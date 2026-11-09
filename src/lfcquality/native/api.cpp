#include "lfcquality/lfcquality.h"
#include "kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct lfq_result {
    lfq_summary summary{};
    lfq_spectral_summary spectrum{};
    std::vector<lfq_acf_value> acf;
    std::vector<lfq_spectral_bin> bins;
};

namespace {
thread_local std::string api_error;
constexpr double undefined = std::numeric_limits<double>::quiet_NaN();
using stats_type = std::array<double, 18>;

void require(bool condition, const char *message) {
    if (!condition) throw std::invalid_argument(message);
}

template<class F> int guarded(F operation) {
    try { operation(); api_error.clear(); return 0; }
    catch (const std::exception &error) { api_error = error.what(); return -1; }
    catch (...) { api_error = "unknown analysis error"; return -1; }
}

void native_call(int status) {
    if (status != 0) throw std::runtime_error(lfq_last_error());
}

int64_t validate_field(const lfq_field *field) {
    require(field != nullptr && field->data != nullptr, "field and data must not be NULL");
    require(field->dtype == LFQ_FLOAT32 || field->dtype == LFQ_FLOAT64, "dtype must be LFQ_FLOAT32 or LFQ_FLOAT64");
    require(field->ndim >= 1 && field->ndim <= 3, "field must have 1, 2 or 3 dimensions");
    const size_t alignment = field->dtype == LFQ_FLOAT32 ? alignof(float) : alignof(double);
    require(reinterpret_cast<uintptr_t>(field->data) % alignment == 0, "data pointer is not naturally aligned");
    int64_t size = 1;
    for (int axis = 0; axis < field->ndim; ++axis) {
        require(field->shape[axis] > 0 && size <= INT64_MAX / field->shape[axis], "invalid or overflowing shape");
        size *= field->shape[axis];
    }
    require(uint64_t(size) <= std::numeric_limits<size_t>::max(), "shape exceeds addressable size");
    require(field->count == size_t(size), "field count must equal the product of shape dimensions");
    return size;
}

bool same_shape(const lfq_field &a, const lfq_field &b) {
    if (a.ndim != b.ndim) return false;
    for (int i = 0; i < a.ndim; ++i) if (a.shape[i] != b.shape[i]) return false;
    return true;
}

int64_t validate_pair(const lfq_field *original, const lfq_field *reconstructed) {
    const int64_t n = validate_field(original);
    validate_field(reconstructed);
    require(same_shape(*original, *reconstructed), "original and reconstruction shapes must match");
    require(original->dtype == reconstructed->dtype, "original and reconstruction dtypes must match");
    return n;
}

void validate_options(const lfq_options &o, const lfq_field &field) {
    require(o.struct_size == sizeof(lfq_options), "options were not initialized with lfq_default_options or have an incompatible ABI");
    require((o.metrics & ~uint32_t(LFQ_METRIC_ALL)) == 0, "unknown metric flags");
    require(o.threads > 0 && o.fft_threads > 0, "thread counts must be positive");
    require(o.threads == 1 || lfq_openmp() != 0, "threads > 1 requires an OpenMP build");
    require(std::isfinite(o.data_range) && o.data_range >= 0, "data_range must be zero (automatic) or finite and positive");
    require(o.ssim_window > 0 && o.ssim_stride > 0, "SSIM window and stride must be positive");
    require(o.boundary == LFQ_BOUNDARY_VALID || o.boundary == LFQ_BOUNDARY_PERIODIC, "invalid ACF boundary");
    require((o.lags == nullptr) == (o.lag_count == 0), "lags and lag_count must both be provided, or both omitted");
    require(o.spectral_bins > 0 && o.spectral_bins <= 65536, "spectral_bins must be between 1 and 65536");
    require(o.spectral_demean == 0 || o.spectral_demean == 1, "spectral_demean must be 0 or 1");
    double radius = 0;
    for (int axis = 0; axis < field.ndim; ++axis) {
        const double spacing = o.spacing[axis];
        require(std::isfinite(spacing) && spacing > 0, "spacing must be finite and positive on every axis");
        radius = std::hypot(radius, double(field.shape[axis] / 2) / field.shape[axis] / spacing);
    }
    require(std::isfinite(radius) && (radius > 0 || field.count == 1), "frequency range overflow or underflow");
    if (o.metrics & LFQ_METRIC_ACF) {
        std::unordered_set<int64_t> seen;
        for (size_t i = 0; i < o.lag_count; ++i) {
            require(o.lags[i] >= 0, "lags must be nonnegative");
            require(seen.insert(o.lags[i]).second, "lags must be unique");
        }
    }
}

stats_type calculate_stats(const lfq_field &a, const lfq_field &b, int threads) {
    stats_type stats{};
    native_call(lfq_stats(a.data, b.data, int(a.dtype), int64_t(a.count), threads, stats.data()));
    require(!(stats[13] == 0 && stats[14] != 0), "RMSE underflowed float64; rescale the inputs");
    return stats;
}

lfq_status variance_status(double value, double variance) {
    if (variance == 0) return LFQ_STATUS_ZERO_VARIANCE;
    return std::isfinite(value) ? LFQ_STATUS_OK : LFQ_STATUS_NUMERIC_RANGE;
}

uint64_t checked_multiply(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("FFT buffer size overflow");
    return a * b;
}
uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("FFT buffer size overflow");
    return a + b;
}
uint64_t fft_buffer_bytes(const lfq_field &field, int bins) {
    const uint64_t n = field.count, last = uint64_t(field.shape[field.ndim-1]);
    const uint64_t m = checked_multiply(n / last, last / 2 + 1);
    const uint64_t blocks = (m-1) / 65536 + 1;
    const uint64_t partial = checked_multiply(checked_multiply(blocks, 6 + 3 * uint64_t(bins)), 8);
    return checked_add(checked_add(checked_multiply(n, 8), checked_multiply(m, 32)), partial);
}

void calculate_acf(lfq_result &result, const lfq_field &x, const lfq_field &y,
                   const lfq_options &options, const stats_type &stats) {
    static constexpr int64_t default_lags[] = {1, 2, 4, 8};
    const int64_t *lags = options.lags ? options.lags : default_lags;
    const size_t count = options.lags ? options.lag_count : sizeof(default_lags)/sizeof(default_lags[0]);
    require(count <= result.acf.max_size() / size_t(x.ndim), "too many ACF lags");
    result.acf.reserve(count * size_t(x.ndim));
    for (int axis = 0; axis < x.ndim; ++axis) {
        const uint64_t length = uint64_t(x.shape[axis]);
        for (size_t k = 0; k < count; ++k) {
            const int64_t lag = lags[k];
            lfq_acf_value row{};
            row.axis = axis; row.lag = lag;
            row.original = row.reconstructed = row.error = row.delta = undefined;
            if (uint64_t(lag) >= length) {
                row.original_status = row.reconstructed_status = row.error_status = LFQ_STATUS_LAG_OUT_OF_RANGE;
            } else {
                row.pairs = options.boundary == LFQ_BOUNDARY_PERIODIC ? x.count : (x.count / length) * (length - uint64_t(lag));
                double values[3];
                native_call(lfq_acf(x.data, y.data, int(x.dtype), x.shape, x.ndim, axis, lag,
                                    options.boundary == LFQ_BOUNDARY_PERIODIC, stats.data(), options.threads, values));
                row.original = values[0]; row.reconstructed = values[1]; row.error = values[2];
                row.original_status = variance_status(values[0], stats[9]);
                row.reconstructed_status = variance_status(values[1], stats[10]);
                row.error_status = variance_status(values[2], stats[11]);
                if (row.original_status != LFQ_STATUS_OK) row.original = undefined;
                if (row.reconstructed_status != LFQ_STATUS_OK) row.reconstructed = undefined;
                if (row.error_status != LFQ_STATUS_OK) row.error = undefined;
                if (row.original_status == LFQ_STATUS_OK && row.reconstructed_status == LFQ_STATUS_OK)
                    row.delta = row.reconstructed - row.original;
            }
            result.acf.push_back(row);
        }
    }
}

void calculate_spectrum(lfq_result &result, const lfq_field &x, const lfq_field &y,
                        const lfq_options &options, const stats_type &stats) {
    require(lfq_fftw() != 0, "spectral analysis requires a library built with FFTW");
    require(options.fft_threads == 1 || lfq_fftw_threads() != 0, "multiple FFT threads require a threaded FFTW build");
    const uint64_t estimate = fft_buffer_bytes(x, options.spectral_bins);
    require(options.max_fft_bytes == 0 || estimate <= options.max_fft_bytes,
            "FFT primary buffers exceed max_fft_bytes");
    std::vector<double> shells(size_t(options.spectral_bins) * 3);
    double totals[10];
    native_call(lfq_spectral_fields(x.data, y.data, int(x.dtype), x.shape, x.ndim, stats.data(), options.spacing,
                                   options.spectral_bins, options.spectral_demean, options.threads, options.fft_threads,
                                   totals, shells.data()));
    for (int i = 0; i < 7; ++i) require(std::isfinite(totals[i]), "non-finite spectral intermediate");
    auto &s = result.spectrum;
    s.amplitude_scale = totals[6]; s.original_power = totals[0]; s.reconstructed_power = totals[1];
    s.absolute_power_l1 = totals[2];
    s.power_relative_l1 = std::isfinite(totals[7]) ? totals[7] : undefined;
    s.power_relative_l2 = std::isfinite(totals[8]) ? totals[8] : undefined;
    s.fourier_relative_l2 = std::isfinite(totals[9]) ? totals[9] : undefined;
    s.primary_buffer_bytes = estimate;
    s.status = totals[0] == 0 ? LFQ_STATUS_ZERO_REFERENCE_POWER :
               (!std::isfinite(totals[7]) || !std::isfinite(totals[8]) || !std::isfinite(totals[9])) ?
               LFQ_STATUS_NUMERIC_RANGE : LFQ_STATUS_OK;
    double rmax = 0;
    for (int axis = 0; axis < x.ndim; ++axis)
        rmax = std::hypot(rmax, double(x.shape[axis]/2) / x.shape[axis] / options.spacing[axis]);
    result.bins.reserve(size_t(options.spectral_bins));
    for (int i = 0; i < options.spectral_bins; ++i) {
        lfq_spectral_bin row{};
        row.lower_frequency = rmax * (double(i) / options.spectral_bins);
        row.upper_frequency = rmax * (double(i+1) / options.spectral_bins);
        row.original_power = shells[i]; row.reconstructed_power = shells[options.spectral_bins+i];
        row.mode_count = uint64_t(shells[2*options.spectral_bins+i]);
        row.relative_delta = undefined;
        if (row.mode_count == 0) row.status = LFQ_STATUS_EMPTY_BIN;
        else if (row.original_power == 0) row.status = LFQ_STATUS_ZERO_REFERENCE_POWER;
        else {
            const double delta = (row.reconstructed_power-row.original_power)/row.original_power;
            row.status = std::isfinite(delta) ? LFQ_STATUS_OK : LFQ_STATUS_NUMERIC_RANGE;
            if (row.status == LFQ_STATUS_OK) row.relative_delta = delta;
        }
        result.bins.push_back(row);
    }
}

std::array<double,4> moments(const stats_type &stats, int kind) {
    return {stats[kind], stats[3+kind], stats[6+kind], stats[9+kind]};
}
}

extern "C" {
int lfq_c_api_version(void) { return LFQ_C_API_VERSION; }
int lfq_openmp_version(void) { return lfq_openmp(); }
int lfq_has_fftw(void) { return lfq_fftw(); }
int lfq_has_threaded_fftw(void) { return lfq_fftw_threads(); }
int lfq_observed_threads(int requested) { return lfq_team_size(requested); }
const char *lfq_fftw_library_version(void) { return lfq_fftw_version(); }

lfq_options lfq_default_options(void) {
    lfq_options options{};
    options.struct_size = sizeof(lfq_options); options.metrics = LFQ_METRIC_ALL;
    options.threads = options.fft_threads = 1;
    options.ssim_window = 7; options.ssim_stride = 4;
    options.boundary = LFQ_BOUNDARY_VALID;
    options.spectral_bins = 64; options.spectral_demean = 1;
    std::fill(std::begin(options.spacing), std::end(options.spacing), 1.0);
    return options;
}

const char *lfq_error_message(void) { return api_error.c_str(); }

int lfq_analyze(const lfq_field *original, const lfq_field *reconstructed,
                const lfq_options *options, lfq_result **out) {
    if (out) *out = nullptr;
    return guarded([&] {
        require(out != nullptr, "output result pointer must not be NULL");
        validate_pair(original, reconstructed);
        const lfq_options defaults = lfq_default_options();
        const lfq_options &o = options ? *options : defaults;
        validate_options(o, *original);
        const stats_type stats = calculate_stats(*original, *reconstructed, o.threads);
        const double range = o.data_range > 0 ? o.data_range : stats[16]-stats[0];
        double basic[4];
        native_call(lfq_basic(stats.data(), range, basic));
        auto result = std::make_unique<lfq_result>();
        auto &s = result->summary;
        s.elements = original->count; s.original_min = stats[0]; s.original_max = stats[16]; s.data_range = range;
        s.rmse = stats[13]; s.max_abs_error = stats[14]; s.mean_error = stats[15];
        s.nrmse = basic[0]; s.psnr_db = basic[1]; s.pcc = basic[2]; s.one_minus_pcc = basic[3];
        s.nrmse_status = range == 0 ? LFQ_STATUS_ZERO_RANGE : std::isinf(basic[0]) ? LFQ_STATUS_POSITIVE_INFINITY :
                         std::isfinite(basic[0]) ? LFQ_STATUS_OK : LFQ_STATUS_NUMERIC_RANGE;
        s.psnr_status = range == 0 ? LFQ_STATUS_ZERO_RANGE : stats[13] == 0 ? LFQ_STATUS_POSITIVE_INFINITY :
                        std::isfinite(basic[1]) ? LFQ_STATUS_OK : LFQ_STATUS_NUMERIC_RANGE;
        s.pcc_status = stats[9] == 0 || stats[10] == 0 ? LFQ_STATUS_ZERO_VARIANCE : variance_status(basic[2], 1);
        s.ssim = undefined; s.ssim_status = LFQ_STATUS_NOT_REQUESTED;
        result->spectrum.amplitude_scale = result->spectrum.original_power = result->spectrum.reconstructed_power = undefined;
        result->spectrum.absolute_power_l1 = result->spectrum.power_relative_l1 = result->spectrum.power_relative_l2 = undefined;
        result->spectrum.fourier_relative_l2 = undefined;
        result->spectrum.status = LFQ_STATUS_NOT_REQUESTED;
        if (o.metrics & LFQ_METRIC_SSIM) {
            bool fits = true;
            for (int i = 0; i < original->ndim; ++i) fits = fits && original->shape[i] >= o.ssim_window;
            if (!fits) s.ssim_status = LFQ_STATUS_WINDOW_TOO_LARGE;
            else if (range == 0) s.ssim_status = LFQ_STATUS_ZERO_RANGE;
            else {
                double values[2];
                native_call(lfq_ssim(original->data, reconstructed->data, int(original->dtype), original->shape,
                                     original->ndim, o.ssim_window, o.ssim_stride, range, stats.data(), o.threads, values));
                require(std::isfinite(values[0]), "non-finite SSIM intermediate");
                s.ssim = values[0]; s.ssim_windows = uint64_t(values[1]); s.ssim_status = LFQ_STATUS_OK;
            }
        }
        if (o.metrics & LFQ_METRIC_ACF) calculate_acf(*result, *original, *reconstructed, o, stats);
        if (o.metrics & LFQ_METRIC_SPECTRUM) calculate_spectrum(*result, *original, *reconstructed, o, stats);
        *out = result.release();
    });
}

void lfq_result_free(lfq_result *result) { delete result; }
const lfq_summary *lfq_result_summary(const lfq_result *result) { return result ? &result->summary : nullptr; }
const lfq_spectral_summary *lfq_result_spectrum(const lfq_result *result) { return result ? &result->spectrum : nullptr; }
const lfq_acf_value *lfq_result_acf(const lfq_result *result, size_t *count) {
    if (count) *count = result ? result->acf.size() : 0;
    return result && !result->acf.empty() ? result->acf.data() : nullptr;
}
const lfq_spectral_bin *lfq_result_spectral_bins(const lfq_result *result, size_t *count) {
    if (count) *count = result ? result->bins.size() : 0;
    return result && !result->bins.empty() ? result->bins.data() : nullptr;
}

int lfq_correlation_matrix(const lfq_field *originals, const lfq_field *reconstructions,
                           size_t field_count, int threads, lfq_correlation *out, size_t out_count) {
    return guarded([&] {
        require(originals && reconstructions && out, "field arrays and output must not be NULL");
        require(field_count >= 2, "at least two aligned variables are required");
        require(field_count <= std::numeric_limits<size_t>::max()/field_count, "matrix dimensions overflow");
        require(out_count >= field_count*field_count, "correlation output buffer is too small");
        require(threads > 0 && (threads == 1 || lfq_openmp() != 0), "invalid thread count or library lacks OpenMP");
        std::vector<stats_type> stats;
        stats.reserve(field_count);
        for (size_t i = 0; i < field_count; ++i) {
            validate_pair(&originals[i], &reconstructions[i]);
            require(same_shape(originals[0], originals[i]), "all variables must have the same grid shape");
            stats.push_back(calculate_stats(originals[i], reconstructions[i], threads));
        }
        for (size_t i = 0; i < field_count; ++i) {
            for (size_t j = i; j < field_count; ++j) {
                lfq_correlation row{};
                double values[2] = {undefined, undefined};
                for (int kind = 0; kind < 2; ++kind) {
                    const auto left = moments(stats[i], kind), right = moments(stats[j], kind);
                    if (i == j) {
                        values[kind] = left[3] > 0 ? 1.0 : undefined;
                    } else {
                        const lfq_field &a = kind == 0 ? originals[i] : reconstructions[i];
                        const lfq_field &b = kind == 0 ? originals[j] : reconstructions[j];
                        native_call(lfq_cross(a.data, int(a.dtype), b.data, int(b.dtype), int64_t(a.count),
                                              left.data(), right.data(), threads, &values[kind]));
                    }
                }
                row.original_pcc = values[0]; row.reconstructed_pcc = values[1];
                row.original_status = stats[i][9] == 0 || stats[j][9] == 0 ? LFQ_STATUS_ZERO_VARIANCE : variance_status(values[0], 1);
                row.reconstructed_status = stats[i][10] == 0 || stats[j][10] == 0 ? LFQ_STATUS_ZERO_VARIANCE : variance_status(values[1], 1);
                if (row.original_status != LFQ_STATUS_OK) row.original_pcc = undefined;
                if (row.reconstructed_status != LFQ_STATUS_OK) row.reconstructed_pcc = undefined;
                row.delta = row.original_status == LFQ_STATUS_OK && row.reconstructed_status == LFQ_STATUS_OK ? values[1]-values[0] : undefined;
                out[i*field_count+j] = out[j*field_count+i] = row;
            }
        }
    });
}
}
