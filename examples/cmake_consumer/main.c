#include <lfcquality/lfcquality.h>
#include <stdio.h>

int main(void) {
    float original[64], reconstructed[64];
    for (int i=0;i<64;++i) { original[i]=(float)i; reconstructed[i]=original[i]+0.01f*(i%3); }
    const lfq_field x = {original,64,LFQ_FLOAT32,2,{8,8}};
    const lfq_field y = {reconstructed,64,LFQ_FLOAT32,2,{8,8}};
    lfq_options options = lfq_default_options();
    options.metrics = 0; /* Request only basic error, PSNR and PCC. */
    lfq_result *result = NULL;
    if (lfq_analyze(&x,&y,&options,&result) != 0) {
        fprintf(stderr,"%s\n",lfq_error_message());
        return 1;
    }
    const lfq_summary *summary = lfq_result_summary(result);
    printf("PSNR = %.4f dB, PCC = %.9f\n",summary->psnr_db,summary->pcc);
    lfq_result_free(result);
    return 0;
}
