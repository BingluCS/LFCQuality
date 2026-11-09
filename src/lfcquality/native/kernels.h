// Independent implementation of published mathematical definitions. No QCAT code.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// C-order, finite native-endian f32/f64 buffers, lengths validated by the caller.
// Return 0 on success, -1 on error; lfq_last_error() is thread-local.
// stats: min x/y/e (0:3), scale x/y/e (3:6), normalized means (6:9),
// population normalized variances (9:12), covariance xy (12), RMSE (13),
// max error (14), mean error (15), max x/y (16:18).
int lfq_abi_version(void);
int lfq_openmp(void);
int lfq_fftw(void);
int lfq_fftw_threads(void);
const char* lfq_fftw_version(void);
int lfq_team_size(int requested);
const char* lfq_last_error(void);
int lfq_stats(const void* x, const void* y, int bytes, int64_t n, int threads, double* out);
// basic[4] = NRMSE, PSNR dB, PCC, 1-PCC; NaN is undefined, +Inf is lossless PSNR.
int lfq_basic(const double* stats, double data_range, double* out);
// acf[3] = original, reconstruction, error; undefined (constant) entries are NaN.
int lfq_acf(const void* x, const void* y, int bytes, const int64_t* shape, int ndim,
            int axis, int64_t lag, int periodic, const double* stats, int threads, double* out);
// ssim[2] = mean score, number of windows. Valid uniform population windows.
int lfq_ssim(const void* x, const void* y, int bytes, const int64_t* shape, int ndim,
             int window, int stride, double data_range, const double* stats, int threads, double* out);
// xm/ym[4] = minimum, scale, normalized mean, normalized population variance.
int lfq_cross(const void* x, int xbytes, const void* y, int ybytes, int64_t n,
              const double* xm, const double* ym, int threads, double* out);
int lfq_prepare_fft(const void* x, int bytes, int64_t n, double lo, double scale,
                    double mean_scaled, int demean, int threads, double* out);
// Interleaved complex128, rfftn shape; bins contain integrated (not averaged) power.
int lfq_spectrum(const double* x, const double* y, const int64_t* shape, int ndim,
                 const double* spacing, int bins, int threads, double* totals, double* shells);
// spectral totals[10]: Px sum, Py sum, |Py-Px| sum, (Py-Px)^2 sum,
// Px^2 sum, |Fy-Fx|^2 sum, amplitude scale, relative power L1,
// relative power L2, relative Fourier L2. shells[3*bins]: Px, Py, mode count.
int lfq_spectral_fields(const void* x, const void* y, int bytes, const int64_t* shape, int ndim,
                        const double* stats, const double* spacing, int bins, int demean,
                        int threads, int fft_threads, double* totals, double* shells);
#ifdef __cplusplus
}
#endif
