// Standalone C++17 example. The Result object releases the C allocation.
#include <lfcquality/lfcquality.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    try {
        constexpr int h=32, w=40, n=h*w;
        std::vector<float> density(n), density_hat(n);
        std::vector<double> temperature(n), temperature_hat(n); // mixed variable dtypes are allowed
        for (int i=0;i<n;++i) {
            density[i] = float(std::sin(i*.03)+.2*std::cos(i*.11));
            density_hat[i] = density[i]+float(.001*std::sin(i*.7));
            temperature[i] = .8*density[i]+.1*std::cos(i*.04);
            temperature_hat[i] = temperature[i]+.002*std::cos(i*.4);
        }
        const auto x = lfcquality::field(density.data(),density.size(),{h,w});
        const auto y = lfcquality::field(density_hat.data(),density_hat.size(),{h,w});
        const auto t = lfcquality::field(temperature.data(),temperature.size(),{h,w});
        const auto th = lfcquality::field(temperature_hat.data(),temperature_hat.size(),{h,w});
        auto options = lfcquality::default_options();
        options.threads = lfq_openmp_version() ? 2 : 1;
        options.fft_threads = 1;
        if (!lfq_has_fftw()) options.metrics &= ~LFQ_METRIC_SPECTRUM;
        const auto result = lfcquality::analyze(x,y,options);
        const auto &summary = result.summary();
        std::cout << "PSNR " << summary.psnr_db << " dB, PCC " << summary.pcc
                  << ", SSIM " << summary.ssim << '\n';
        const auto acf = result.autocorrelation();
        if (!acf.empty() && acf[0].error_status == LFQ_STATUS_OK)
            std::cout << "error ACF " << acf[0].error << ", field ACF change " << acf[0].delta << '\n';
        const auto bins = result.spectral_bins();
        if (!bins.empty()) std::cout << "spectral relative power L1 " << result.spectrum().power_relative_l1 << '\n';
        const auto matrix = lfcquality::correlation_matrix({x,t},{y,th},options.threads);
        std::cout << "density/temperature PCC change " << matrix[1].delta << '\n';
        return summary.psnr_status == LFQ_STATUS_OK && summary.ssim_status == LFQ_STATUS_OK &&
               matrix[1].original_status == LFQ_STATUS_OK ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "LFCQuality: " << error.what() << '\n';
        return 1;
    }
}
