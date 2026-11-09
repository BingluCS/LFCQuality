# HPDPS-8 validation — 2026-11-09

Final acceptance was run on HPDPS-8 (Intel Xeon Gold 6530, Linux x86-64),
bound to physical CPUs 32–39 and NUMA node 1. The existing SDRBench run was
active on CPU 2 on the other socket. These performance numbers are initial
observations with that concurrent workload, not dedicated-machine measurements.

## Correctness and integration

- **56 tests passed, zero skipped**: independent NumPy/skimage references,
  error and field ACF boundaries, 1D/2D/3D SSIM, full FFT vs half-spectrum
  multiplicities, odd/even sizes, Parseval, degenerate inputs, thread consistency,
  CLI file handling, and the existing LFCBenchmark tests including SZ3 and ZFP.
- **1/1 CTest passed**: a pure C++ executable using the public native API.
- **Installed wheel passed** in a separate fresh environment with only NumPy
  as a Python runtime dependency. Its native C++ library was rebuilt from the
  packaged sources. All metric groups ran without importing lfcbench, SciPy or
  scikit-image.
- The lfcbench registration patch applies cleanly to `da5f4d0` and produces
  the exact source files used by the integration tests.
- Two fields each from NYX float32 and Miranda float64 were tested using
  leading 128³ crops, ZFP fixed-accuracy reconstructions and alpha=1E-3 of the
  original range. All error bounds passed. All metric groups, including the
  cross-variable matrix, agreed exactly at 1 and 8 OpenMP threads on the same
  native binary. This is an integration check on crops, not a full dataset study.

Evidence: [pytest](pytest.txt), [C++](ctest.txt), [wheel](wheel-check.json),
[patch](patch-check.txt), [real-data summary](sdrbench-smoke/summary.json),
[NYX](sdrbench-smoke/NYX.json), [Miranda](sdrbench-smoke/Miranda.json).

## OpenMP timing observation

128³ float32 deterministic synthetic fields, one warmup and three measurements
per case, randomized case order. Times include the Python API and the mandatory
basic scans for each selected metric group. They exclude input creation and
compressor calls. FFTW was fixed to one thread to measure the surrounding OpenMP
work; therefore the spectral speedup does not include parallel FFT execution.

| Analysis group | 1 thread (ms) | 8 threads (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Basic statistics / PSNR / PCC | 23.744 | 4.588 | 5.18× |
| Error and field ACF | 213.954 | 35.657 | 6.00× |
| SSIM | 85.972 | 14.697 | 5.85× |
| Spectrum, FFTW threads fixed at 1 | 107.981 | 49.890 | 2.16× |

[Raw timings](openmp-128.json) include all measurements and 1/2/4/8-thread
cases. `cpu_affinity` in that report is the main thread's mask **at the end**
of the run, after OpenMP binding; [numa-binding.txt](numa-binding.txt) records
the launch binding for the full analysis process. Team-size probes confirm the
requested worker counts. [Environment](environment.json), [native build](native-build.json)
and [native linkage](native-linkage.txt) identify the tested toolchain.

The CMake configure step noted that the isolated FFTW dependency prefix also
contains libgomp/libstdc++; the recorded linkage resolves those runtimes from
that prefix. The C++ and Python checks above used this configuration.
