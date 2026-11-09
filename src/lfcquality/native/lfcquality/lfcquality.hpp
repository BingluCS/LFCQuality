#ifndef LFCQUALITY_LFCQUALITY_HPP
#define LFCQUALITY_LFCQUALITY_HPP

#include <lfcquality/lfcquality.h>

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace lfcquality {

/* Non-owning view. Keep data alive and unchanged while a call is running. */
template<class T>
lfq_field field(const T *data, std::size_t count, std::initializer_list<int64_t> shape) {
    using Scalar = std::remove_cv_t<T>;
    static_assert(std::is_same_v<Scalar,float> || std::is_same_v<Scalar,double>, "T must be float or double");
    if (shape.size() < 1 || shape.size() > 3) throw std::invalid_argument("shape must have 1, 2 or 3 dimensions");
    lfq_field result{};
    result.data = data;
    result.count = count;
    result.dtype = std::is_same_v<Scalar,float> ? LFQ_FLOAT32 : LFQ_FLOAT64;
    result.ndim = static_cast<int>(shape.size());
    std::copy(shape.begin(), shape.end(), result.shape);
    return result;
}

inline lfq_options default_options() { return lfq_default_options(); }

class Result {
public:
    explicit Result(lfq_result *result) : result_(result, &lfq_result_free) {
        if (!result) throw std::invalid_argument("result must not be null");
    }
    Result(Result &&) noexcept = default;
    Result &operator=(Result &&) noexcept = default;
    Result(const Result &) = delete;
    Result &operator=(const Result &) = delete;

    const lfq_summary &summary() const { return *lfq_result_summary(result_.get()); }
    const lfq_spectral_summary &spectrum() const { return *lfq_result_spectrum(result_.get()); }
    std::vector<lfq_acf_value> autocorrelation() const {
        std::size_t count = 0;
        const auto *values = lfq_result_acf(result_.get(), &count);
        return count ? std::vector<lfq_acf_value>(values, values+count) : std::vector<lfq_acf_value>{};
    }
    std::vector<lfq_spectral_bin> spectral_bins() const {
        std::size_t count = 0;
        const auto *values = lfq_result_spectral_bins(result_.get(), &count);
        return count ? std::vector<lfq_spectral_bin>(values, values+count) : std::vector<lfq_spectral_bin>{};
    }

private:
    std::unique_ptr<lfq_result, decltype(&lfq_result_free)> result_;
};

inline Result analyze(const lfq_field &original, const lfq_field &reconstructed,
                      const lfq_options &options = default_options()) {
    lfq_result *result = nullptr;
    if (lfq_analyze(&original, &reconstructed, &options, &result) != 0)
        throw std::runtime_error(lfq_error_message());
    return Result(result);
}

/* Returns an N x N row-major matrix; the caller ensures spatial/time alignment. */
inline std::vector<lfq_correlation> correlation_matrix(const std::vector<lfq_field> &originals,
                                                       const std::vector<lfq_field> &reconstructions,
                                                       int threads = 1) {
    const std::size_t n = originals.size();
    if (n != reconstructions.size() || n < 2 || n > std::numeric_limits<std::size_t>::max()/n)
        throw std::invalid_argument("at least two matching fields are required");
    std::vector<lfq_correlation> result(n*n);
    if (lfq_correlation_matrix(originals.data(), reconstructions.data(), n, threads, result.data(), result.size()) != 0)
        throw std::runtime_error(lfq_error_message());
    return result;
}
}
#endif
