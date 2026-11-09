/* Standalone C11 example: no Python or NumPy is required. */
#include <lfcquality/lfcquality.h>
#include <math.h>
#include <stdio.h>

int main(void) {
    enum { H = 32, W = 40, N = H * W };
    float density[N], density_hat[N], temperature[N], temperature_hat[N];
    for (int i = 0; i < N; ++i) {
        density[i] = (float)(sin(i * 0.03) + 0.2*cos(i * 0.11));
        density_hat[i] = density[i] + (float)(0.001*sin(i * 0.7));
        temperature[i] = 0.8f*density[i] + (float)(0.1*cos(i * 0.04));
        temperature_hat[i] = temperature[i] + (float)(0.002*cos(i * 0.4));
    }
    const lfq_field original[2] = {
        {density, N, LFQ_FLOAT32, 2, {H, W}},
        {temperature, N, LFQ_FLOAT32, 2, {H, W}}
    };
    const lfq_field reconstructed[2] = {
        {density_hat, N, LFQ_FLOAT32, 2, {H, W}},
        {temperature_hat, N, LFQ_FLOAT32, 2, {H, W}}
    };
    lfq_options options = lfq_default_options();
    options.threads = lfq_openmp_version() ? 2 : 1;
    options.fft_threads = 1;
    if (!lfq_has_fftw()) options.metrics &= ~LFQ_METRIC_SPECTRUM;

    lfq_result *result = NULL;
    if (lfq_analyze(&original[0], &reconstructed[0], &options, &result) != 0) {
        fprintf(stderr, "analysis failed: %s\n", lfq_error_message());
        return 1;
    }
    const lfq_summary *summary = lfq_result_summary(result);
    printf("PSNR %.3f dB, PCC %.9f, SSIM %.9f\n", summary->psnr_db, summary->pcc, summary->ssim);
    size_t count = 0;
    const lfq_acf_value *acf = lfq_result_acf(result, &count);
    if (count && acf[0].error_status == LFQ_STATUS_OK)
        printf("axis %d lag %lld: error ACF %.6f, field ACF change %.6f\n",
               acf[0].axis, (long long)acf[0].lag, acf[0].error, acf[0].delta);
    const lfq_spectral_summary *spectrum = lfq_result_spectrum(result);
    if (spectrum->status == LFQ_STATUS_OK) {
        const lfq_spectral_bin *bins = lfq_result_spectral_bins(result, &count);
        printf("spectral relative power L1 %.6g (%zu bins, first has %llu modes)\n",
               spectrum->power_relative_l1, count, (unsigned long long)bins[0].mode_count);
    }
    lfq_correlation matrix[4];
    if (lfq_correlation_matrix(original, reconstructed, 2, options.threads, matrix, 4) != 0) {
        fprintf(stderr, "correlation failed: %s\n", lfq_error_message());
        lfq_result_free(result);
        return 1;
    }
    printf("density/temperature PCC change %.6g\n", matrix[1].delta);
    const int ok = summary->psnr_status == LFQ_STATUS_OK && summary->pcc_status == LFQ_STATUS_OK &&
                   summary->ssim_status == LFQ_STATUS_OK && matrix[1].original_status == LFQ_STATUS_OK;
    lfq_result_free(result);
    return ok ? 0 : 1;
}
