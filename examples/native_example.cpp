// Python-free use of the same native library. See kernels.h for output layouts.
#include "kernels.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(int status) { if(status) throw std::runtime_error(lfq_last_error()); }
int main() {
    try {
        const int64_t shape[]={16,16,16};
        constexpr int64_t n=16*16*16;
        std::vector<double> original(n),reconstructed(n);
        for(int64_t i=0;i<n;++i) {original[i]=std::sin(i*.01); reconstructed[i]=original[i]+.001*std::cos(i*.7);}
        const int threads=lfq_openmp()?4:1;
        double stats[18], basic[4], ssim[2], acf[3];
        check(lfq_stats(original.data(),reconstructed.data(),8,n,threads,stats));
        check(lfq_basic(stats,stats[16]-stats[0],basic));
        check(lfq_ssim(original.data(),reconstructed.data(),8,shape,3,7,4,stats[16]-stats[0],stats,threads,ssim));
        check(lfq_acf(original.data(),reconstructed.data(),8,shape,3,2,1,0,stats,threads,acf));
        std::cout<<std::setprecision(12)<<"PSNR="<<basic[1]<<" PCC="<<basic[2]<<" SSIM="<<ssim[0]<<" error_ACF_axis2_lag1="<<acf[2]<<'\n';
        if(lfq_fftw()) {
            const double spacing[]={1,1,1};
            double totals[10],shells[3*16];
            check(lfq_spectral_fields(original.data(),reconstructed.data(),8,shape,3,stats,spacing,16,1,threads,1,totals,shells));
            std::cout<<"Power_relative_L1="<<totals[7]<<'\n';
        }
        return basic[1]>60 && basic[2]>.99 && ssim[0]>.99 ? 0:1;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
