/* Public C API for scientific floating-point reconstruction analysis. */
#ifndef LFCQUALITY_LFCQUALITY_H
#define LFCQUALITY_LFCQUALITY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LFQ_C_API_VERSION 1

/* Data are native-endian, naturally aligned, contiguous C-order arrays. */
typedef enum lfq_dtype {
    LFQ_FLOAT32 = 4,
    LFQ_FLOAT64 = 8
} lfq_dtype;

typedef struct lfq_field {
    const void *data;
    size_t count;             /* Number of elements, not bytes. */
    lfq_dtype dtype;
    int ndim;                 /* 1, 2 or 3. */
    int64_t shape[3];         /* First ndim dimensions, slowest to fastest. */
} lfq_field;

typedef enum lfq_status {
    LFQ_STATUS_OK = 0,
    LFQ_STATUS_NOT_REQUESTED = 1,
    LFQ_STATUS_ZERO_RANGE = 2,
    LFQ_STATUS_ZERO_VARIANCE = 3,
    LFQ_STATUS_LAG_OUT_OF_RANGE = 4,
    LFQ_STATUS_WINDOW_TOO_LARGE = 5,
    LFQ_STATUS_POSITIVE_INFINITY = 6,
    LFQ_STATUS_ZERO_REFERENCE_POWER = 7,
    LFQ_STATUS_EMPTY_BIN = 8,
    LFQ_STATUS_NUMERIC_RANGE = 9
} lfq_status;

typedef enum lfq_boundary {
    LFQ_BOUNDARY_VALID = 0,
    LFQ_BOUNDARY_PERIODIC = 1
} lfq_boundary;

/* Basic error, PSNR and PCC are always computed. These flags add other work. */
enum {
    LFQ_METRIC_SSIM = 1u << 0,
    LFQ_METRIC_ACF = 1u << 1,
    LFQ_METRIC_SPECTRUM = 1u << 2,
    LFQ_METRIC_ALL = LFQ_METRIC_SSIM | LFQ_METRIC_ACF | LFQ_METRIC_SPECTRUM
};

typedef struct lfq_options {
    size_t struct_size;       /* Set by lfq_default_options(); do not change. */
    uint32_t metrics;         /* LFQ_METRIC_* bit mask. */
    int threads;              /* OpenMP threads for the analysis kernels. */
    int fft_threads;          /* FFTW threads, controlled separately. */
    double data_range;        /* 0: original max-min; otherwise positive. */
    int ssim_window;
    int ssim_stride;
    lfq_boundary boundary;    /* Applies to ACF; SSIM always uses valid windows. */
    const int64_t *lags;      /* Borrowed for the duration of lfq_analyze(). */
    size_t lag_count;         /* 0 with NULL lags selects 1, 2, 4, 8. */
    int spectral_bins;
    int spectral_demean;      /* 0 or 1. */
    double spacing[3];        /* First ndim spacings, in one common unit. */
    uint64_t max_fft_bytes;   /* 0: no limit; primary buffers only, not RSS. */
} lfq_options;

typedef struct lfq_summary {
    size_t elements;
    double original_min;
    double original_max;
    double data_range;
    double max_abs_error;
    double mean_error;        /* reconstructed - original */
    double rmse;
    double nrmse;
    double psnr_db;
    double pcc;
    double one_minus_pcc;
    double ssim;
    uint64_t ssim_windows;
    lfq_status nrmse_status;
    lfq_status psnr_status;
    lfq_status pcc_status;
    lfq_status ssim_status;
} lfq_summary;

typedef struct lfq_acf_value {
    int axis;                 /* Zero-based; follows the supplied shape order. */
    int64_t lag;
    uint64_t pairs;
    double original;
    double reconstructed;
    double error;
    double delta;             /* reconstructed ACF - original ACF */
    lfq_status original_status;
    lfq_status reconstructed_status;
    lfq_status error_status;
} lfq_acf_value;

typedef struct lfq_spectral_summary {
    double amplitude_scale;
    double original_power;
    double reconstructed_power;
    double absolute_power_l1;
    double power_relative_l1;
    double power_relative_l2;
    double fourier_relative_l2;
    uint64_t primary_buffer_bytes;
    lfq_status status;
} lfq_spectral_summary;

typedef struct lfq_spectral_bin {
    double lower_frequency;
    double upper_frequency;
    uint64_t mode_count;      /* Counts the full grid, including conjugates. */
    double original_power;
    double reconstructed_power;
    double relative_delta;
    lfq_status status;
} lfq_spectral_bin;

typedef struct lfq_correlation {
    double original_pcc;
    double reconstructed_pcc;
    double delta;
    lfq_status original_status;
    lfq_status reconstructed_status;
} lfq_correlation;

typedef struct lfq_result lfq_result; /* Opaque; use accessors and free below. */

int lfq_c_api_version(void);
int lfq_openmp_version(void);      /* 0 when OpenMP is unavailable. */
int lfq_has_fftw(void);
int lfq_has_threaded_fftw(void);
int lfq_observed_threads(int requested); /* Probe; OS/runtime may limit a team. */
const char *lfq_fftw_library_version(void);
lfq_options lfq_default_options(void);
const char *lfq_error_message(void); /* Thread-local; valid until next API call. */

/* Returns 0 on success, -1 on failure. *out is NULL on failure. Input buffers
 * are borrowed only until this call returns; result storage is owned by the
 * library and must be released exactly once with lfq_result_free(). */
int lfq_analyze(const lfq_field *original, const lfq_field *reconstructed,
                const lfq_options *options, lfq_result **out);
void lfq_result_free(lfq_result *result); /* NULL is accepted. */

/* Accessors return borrowed pointers valid until lfq_result_free(). For an
 * omitted/empty vector, *count is 0 and NULL may be returned. */
const lfq_summary *lfq_result_summary(const lfq_result *result);
const lfq_acf_value *lfq_result_acf(const lfq_result *result, size_t *count);
const lfq_spectral_summary *lfq_result_spectrum(const lfq_result *result);
const lfq_spectral_bin *lfq_result_spectral_bins(const lfq_result *result, size_t *count);

/* Pairwise cross-variable PCC for co-located fields. Original/reconstructed
 * arrays each have field_count entries. Different fields may use different
 * dtypes, but each original/reconstruction pair must match. out has at least
 * field_count*field_count entries in row-major order. */
int lfq_correlation_matrix(const lfq_field *originals, const lfq_field *reconstructions,
                           size_t field_count, int threads, lfq_correlation *out,
                           size_t out_count);

#ifdef __cplusplus
}
#endif
#endif
