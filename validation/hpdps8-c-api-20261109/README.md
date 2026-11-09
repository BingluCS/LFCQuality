# C/C++ API validation on HPDPS-8

Validation for LFCQuality 0.2.0 ran on HPDPS-8 (Linux x86-64, Intel Xeon Gold
6530) using GCC/G++ 11.4.0, CMake 3.22.1 and FFTW 3.3.11. Tests were bound to
CPUs 32–39 and NUMA node 1. The native-source SHA-256 was
`847d486608630c45a8f6425dc28deef78746defb4d091c311bdda5353b3bc619`;
the installed wheel produced the same source hash.

- [CMake/CTest with OpenMP and FFTW](ctest.log): all five native tests passed,
  including C11 and C++17 examples, C boundary/error cases and C++ RAII moves.
- [CMake/CTest with OpenMP and FFTW disabled](serial-ctest.log): all five tests
  passed, including explicit errors for unavailable features.
- The installed library was linked and executed using `gcc -std=c11` and
  `g++ -std=c++17`: [C output](c-example.log), [C++ output](cpp-example.log).
  A downstream [C-only CMake consumer](consumer.log) found and linked the
  installed `LFCQuality::lfcquality` target. The library has no Python runtime
  dependency.
- Both complete README snippets were compiled and run on the server: [C](readme-c.log)
  and [C++](readme-cpp.log).
- [Combined Python and LFCBenchmark tests](combined-pytest.log): 75 passed.
  These include C ABI comparisons with the Python API for 2D/3D, float32/float64,
  valid/periodic boundaries, SSIM, ACF, spectra, mixed-precision cross-variable
  matrices, undefined values, ownership and thread-local errors.
- The 0.2.0 source distribution and wheel included the public headers and source.
  The wheel was installed in a separate environment, built from its packaged
  sources, and exercised without importing SciPy or LFCBenchmark:
  [wheel check](wheel-check.log).

The [source build metadata](native-build.json) and
[wheel build metadata](wheel-native-build.json) record compiler flags, native
source hashes and shared-library hashes. These checks validate API correctness;
the prior [0.1.0 record](../hpdps8-20261109/README.md) contains the SDRBench
crop experiments and performance observations.
