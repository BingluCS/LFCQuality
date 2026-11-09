#include <lfcquality/lfcquality.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #expr, lfq_error_message()); return 1; } } while (0)

int main(void) {
    float a[] = {0,1,2,3,4,5}, ar[] = {0,1,2,3,4,5};
    double b[] = {1,3,5,7,9,11}, br[] = {1,-1,-3,-5,-7,-9};
    float zero[] = {0,0,0,0,0,0};
    lfq_field original[] = {{a,6,LFQ_FLOAT32,2,{2,3}}, {b,6,LFQ_FLOAT64,2,{2,3}}, {zero,6,LFQ_FLOAT32,2,{2,3}}};
    lfq_field recon[] = {{ar,6,LFQ_FLOAT32,2,{2,3}}, {br,6,LFQ_FLOAT64,2,{2,3}}, {zero,6,LFQ_FLOAT32,2,{2,3}}};
    CHECK(lfq_c_api_version() == LFQ_C_API_VERSION);
    lfq_options options = lfq_default_options();
    CHECK(options.struct_size == sizeof(options));
    options.metrics = LFQ_METRIC_ACF | LFQ_METRIC_SSIM;
    lfq_result *result = NULL;
    CHECK(lfq_analyze(&original[0], &recon[0], &options, &result) == 0);
    const lfq_summary *s = lfq_result_summary(result);
    CHECK(s && s->elements == 6 && s->max_abs_error == 0);
    CHECK(s->psnr_status == LFQ_STATUS_POSITIVE_INFINITY && isinf(s->psnr_db));
    CHECK(s->pcc_status == LFQ_STATUS_OK && fabs(s->pcc-1) < 1e-14);
    CHECK(s->ssim_status == LFQ_STATUS_WINDOW_TOO_LARGE && isnan(s->ssim));
    size_t count = 0;
    const lfq_acf_value *acf = lfq_result_acf(result,&count);
    CHECK(acf && count == 8 && acf[0].axis == 0 && acf[0].lag == 1 && acf[0].pairs == 3);
    CHECK(acf[0].error_status == LFQ_STATUS_ZERO_VARIANCE && isnan(acf[0].error));
    CHECK(acf[1].original_status == LFQ_STATUS_LAG_OUT_OF_RANGE && acf[1].pairs == 0);
    CHECK(lfq_result_spectrum(result)->status == LFQ_STATUS_NOT_REQUESTED && isnan(lfq_result_spectrum(result)->original_power));
    CHECK(lfq_result_spectral_bins(result,&count) == NULL && count == 0);
    lfq_result_free(result); result = NULL;

    lfq_field bad = original[0]; bad.count = 5;
    CHECK(lfq_analyze(&bad,&recon[0],&options,&result) != 0 && result == NULL);
    CHECK(strstr(lfq_error_message(),"count") != NULL);
    const int64_t duplicates[] = {1,1};
    options.lags = duplicates; options.lag_count = 2;
    CHECK(lfq_analyze(&original[0],&recon[0],&options,&result) != 0 && result == NULL);
    CHECK(strstr(lfq_error_message(),"unique") != NULL);
    options = lfq_default_options(); options.metrics = LFQ_METRIC_SSIM | LFQ_METRIC_ACF;
    options.data_range = 1; options.ssim_window = 2; options.ssim_stride = 1;
    CHECK(lfq_analyze(&original[2],&recon[2],&options,&result) == 0);
    s = lfq_result_summary(result);
    CHECK(s->ssim_status == LFQ_STATUS_OK && s->ssim == 1);
    CHECK(s->pcc_status == LFQ_STATUS_ZERO_VARIANCE && isnan(s->pcc));
    lfq_result_free(result); result = NULL;

    lfq_correlation matrix[9];
    CHECK(lfq_correlation_matrix(original,recon,3,1,matrix,8) != 0);
    CHECK(lfq_correlation_matrix(original,recon,3,1,matrix,9) == 0);
    CHECK(fabs(matrix[1].original_pcc-1) < 1e-14 && fabs(matrix[1].reconstructed_pcc+1) < 1e-14);
    CHECK(fabs(matrix[1].delta+2) < 1e-14 && matrix[3].delta == matrix[1].delta);
    CHECK(matrix[2].original_status == LFQ_STATUS_ZERO_VARIANCE && isnan(matrix[2].delta));
    if (lfq_has_fftw()) {
        CHECK(lfq_analyze(&original[0],&recon[0],NULL,&result) == 0);
        lfq_result_free(result); result = NULL;
        options = lfq_default_options(); options.metrics = LFQ_METRIC_SPECTRUM; options.spectral_bins = 4;
        options.max_fft_bytes = 1;
        CHECK(lfq_analyze(&original[0],&recon[0],&options,&result) != 0 && result == NULL);
        CHECK(strstr(lfq_error_message(),"max_fft_bytes") != NULL);
        options.max_fft_bytes = 0;
        CHECK(lfq_analyze(&original[0],&recon[0],&options,&result) == 0);
        const lfq_spectral_summary *spectrum = lfq_result_spectrum(result);
        CHECK(spectrum && spectrum->status == LFQ_STATUS_OK && spectrum->power_relative_l1 == 0);
        const lfq_spectral_bin *bins = lfq_result_spectral_bins(result,&count);
        CHECK(bins && count == 4);
        uint64_t modes = 0;
        for (size_t i=0;i<count;++i) modes += bins[i].mode_count;
        CHECK(modes == 6);
        lfq_result_free(result);
    } else {
        options = lfq_default_options(); result = NULL;
        CHECK(lfq_analyze(&original[0],&recon[0],&options,&result) != 0 && result == NULL);
        CHECK(strstr(lfq_error_message(),"FFTW") != NULL);
    }
    if (!lfq_openmp_version()) {
        options = lfq_default_options(); options.metrics = 0; options.threads = 2; result = NULL;
        CHECK(lfq_analyze(&original[0],&recon[0],&options,&result) != 0 && result == NULL);
        CHECK(strstr(lfq_error_message(),"OpenMP") != NULL);
    }
    lfq_result_free(NULL);
    puts("C API edge cases passed");
    return 0;
}
