#include <lfcquality/lfcquality.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { std::cerr << "FAIL " << __LINE__ << ": " #expr << '\n'; return 1; } } while (0)

int main() {
    std::vector<float> a{0,1,2,3,4,5}, ar{0,1,2,3,4,5};
    std::vector<double> b{1,3,5,7,9,11}, br{1,-1,-3,-5,-7,-9};
    const auto x = lfcquality::field(a.data(),a.size(),{2,3});
    const auto y = lfcquality::field(ar.data(),ar.size(),{2,3});
    const auto u = lfcquality::field(b.data(),b.size(),{2,3});
    const auto v = lfcquality::field(br.data(),br.size(),{2,3});
    auto options = lfcquality::default_options();
    options.metrics = 0;
    auto first = lfcquality::analyze(x,y,options);
    auto moved = std::move(first);
    CHECK(moved.summary().psnr_status == LFQ_STATUS_POSITIVE_INFINITY);
    CHECK(moved.summary().pcc > .999999999999);
    CHECK(moved.autocorrelation().empty() && moved.spectral_bins().empty());
    CHECK(moved.spectrum().status == LFQ_STATUS_NOT_REQUESTED);
    const auto matrix = lfcquality::correlation_matrix({x,u},{y,v});
    CHECK(matrix.size() == 4 && std::abs(matrix[1].delta+2) < 1e-14);
    bool field_error = false, analyze_error = false, matrix_error = false;
    try { (void)lfcquality::field(a.data(),a.size(),{}); }
    catch (const std::invalid_argument &) { field_error = true; }
    try { (void)lfcquality::analyze(x,u,options); }
    catch (const std::runtime_error &e) { analyze_error = std::string(e.what()).find("dtype") != std::string::npos; }
    try { (void)lfcquality::correlation_matrix({x,u},{y}); }
    catch (const std::invalid_argument &) { matrix_error = true; }
    CHECK(field_error && analyze_error && matrix_error);
    std::cout << "C++ RAII and error handling passed\n";
}
