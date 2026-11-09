# LFCQuality

Independent quality analysis of scientific floating-point reconstructions. All
data-sized numerical work runs in C++17, with OpenMP kernels and FFTW transforms.
Python provides the NumPy API, command line and optional LFCBenchmark integration.

Supports PSNR/NRMSE, local 1D/2D/3D SSIM, Pearson correlation, error
autocorrelation, original/reconstructed field autocorrelation changes,
Fourier/power-spectrum errors and cross-variable correlation matrix changes.

This is an independent implementation of published mathematical definitions.
[QCAT](https://github.com/szcompressor/qcat) was consulted for its documented
analysis scope and file-oriented interface; no QCAT source code is copied,
translated, linked or vendored. This package does not promise numerical
equivalence to QCAT's particular metric conventions.

## Install and build

Python 3.10+, a C++17 compiler, OpenMP and FFTW3 (including its threads library)
are needed for all metrics. Python has only NumPy as a runtime dependency.

```bash
# Ubuntu/Debian, if the native dependencies are missing:
sudo apt-get install g++ libfftw3-dev
# macOS alternative:
brew install libomp fftw

python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
lfcquality build --openmp required --fftw required
lfcquality info
```

Builds live in a source/compiler/flag-specific user cache. Importing the package
never launches a compiler. `CXX`, `CPPFLAGS`, `CXXFLAGS`, `LDFLAGS`,
`LFCQUALITY_OMP_PREFIX` and `LFCQUALITY_FFTW_PREFIX` support custom toolchains.
Use `LFCQUALITY_CACHE` to relocate the cache and `LFCQUALITY_LIBRARY` to load a
specific native library. A build with `--openmp off` accepts only `threads=1`;
`--fftw off` omits spectral analysis. Missing capabilities raise errors rather
than silently returning a different implementation.

## Python API

```python
import numpy as np
from lfcquality import analyze, analyze_fields

original = np.load("original.npy", mmap_mode="r")
reconstructed = np.load("reconstructed.npy", mmap_mode="r")
result = analyze(
    original, reconstructed,
    threads=8, fft_threads=4,
    lags=(1, 2, 4, 8), boundary="valid",
    ssim_window=7, ssim_stride=4,
    spacing=(1., 1., 1.), spectral_bins=64,
)
print(result["metrics"]["psnr"])
print(result["metrics"]["pcc"])

# Select inexpensive metrics, e.g. while iterating on a codec:
small = analyze(original, reconstructed, metrics=("psnr", "pcc"), threads=8)

# Supply variables on the SAME spatial grid and time step:
joint = analyze_fields(
    {"density": density, "temperature": temperature},
    {"density": density_reconstructed, "temperature": temperature_reconstructed},
    metrics=("psnr", "pcc", "error_autocorrelation"), threads=8,
)
print(joint["cross_variable_correlation"]["delta"])
```

Inputs must be finite native-endian float32/float64, 1D/2D/3D arrays. The
original/reconstruction pair must match in dtype and shape. C-contiguous aligned
arrays and memmaps pass through without copying; other layouts are copied once.
SSIM uses the full dimensionality (a 3D field uses 3D windows, not image slices).

## Pure C++ use

The native library also builds without Python. Its public C-compatible API is
[`kernels.h`](src/lfcquality/native/kernels.h); see the
[C++ example](examples/native_example.cpp) for in-memory arrays.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
./build/lfcquality_example
ctest --test-dir build --output-on-failure
```

On Homebrew macOS, add
`-DCMAKE_PREFIX_PATH="$(brew --prefix libomp);$(brew --prefix fftw)"` when
configuring CMake. To select this library in Python, set
`LFCQUALITY_LIBRARY` to the absolute path of `build/liblfcquality.so` (Linux)
or `build/liblfcquality.dylib` (macOS). CMake builds report the binary hash;
the Python build command additionally records compiler flags and source hashes.

## Command line

```bash
lfcquality compare original.npy reconstructed.npy --threads 8 --fft-threads 4 \
  --output results/quality.json

# RAW shape is slowest-to-fastest C order: z y x.
lfcquality compare original.f32 reconstructed.f32 \
  --dtype float32 --shape 100 500 500 --threads 8 \
  --output results/hurricane-quality.json

# Independent analysis of aligned variables, relative paths resolved at manifest:
lfcquality compare-fields examples/fields.json --output results/variables.json

# Runnable synthetic demonstration, no external datasets:
python examples/demo.py --output results/demo
```

Outputs are strict JSON, with metric conventions, native source/binary hashes,
build options, requested OpenMP threads and an actual team-size probe. The CLI
also records input SHA-256 hashes. Existing output files are never overwritten.

## Metric conventions and reproducibility

- PSNR uses the original `max-min` by default; `data_range` overrides it.
  NRMSE uses that same range. Zero error has `positive_infinity` PSNR; undefined
  or infinite JSON values are `null` with an explicit status.
- PCC is centered and scaled. A constant input makes it undefined. Pearson 1
  alone does not guarantee preservation of scale or offset.
- SSIM defaults to uniform 7-point windows on every spatial axis, population
  moments, `K1=0.01`, `K2=0.03`, and valid windows at stride 4. **Stride 4 is a
  subsampled window grid.** Set `ssim_stride=1` for all valid windows. Its
  conventions differ from Gaussian-window SSIM; always report the settings.
- ACF uses global means and the full-field centered sum of squares in the
  denominator. Default valid boundaries never connect adjacent flattened rows.
  `periodic` wraps within the selected axis. Undefined lags/constants are marked.
- Spectrum errors use full-dimensional FFTW real transforms, correct Hermitian
  multiplicities (including odd lengths/Nyquist planes), and optional mean
  removal (default on). Full-grid errors and radial shell-integrated powers are
  both returned. Phase-sensitive Fourier error and phase-insensitive power
  error are separate. Spacing must use a common unit on every axis.
- Cross-variable changes are `PCC_reconstructed - PCC_original`, in named field
  order. The summary counts each off-diagonal pair once, excluding undefined
  entries and reporting how many are valid. Matching shapes alone cannot prove
  physical alignment; provide co-located variables.

See [the precise definitions](docs/metrics.md) for formulas, normalization,
degenerate cases and source references.

## Parallelism and memory

OpenMP parallelizes validation/range scans, fixed-block moment reductions,
lag-product accumulation, independent SSIM windows, pairwise field reductions,
FFT input preparation and spectral shell reductions. It uses `num_threads` on
each region; it does not modify compressor thread settings or CPU affinity.
Fixed block boundaries and an ordered merge make results insensitive to OpenMP
iteration scheduling on a fixed binary. FFTW has a separate `fft_threads` setting;
different FFT thread counts/builds can change the last few floating-point bits.

Basic/ACF/SSIM kernels avoid allocating full-field error or variance arrays.
Spectral analysis uses one double real buffer and two complex-double half-spectrum
buffers, approximately **24 bytes per input element**, plus shell reductions,
FFTW scratch and the two inputs. At 512³ that is about 3 GiB of primary FFT
buffers. `max_fft_bytes` is a preflight check for primary buffers, **not** a bound
on total RSS. Use fewer shell bins if needed. No hidden field subsampling is
performed for ACF, PCC or spectra.

SSIM cost grows as the window volume times the number of windows; stride 1 on a
large 3D field can be expensive. Configurable stride makes this cost explicit.

## LFCBenchmark integration

The package advertises the `lfcquality` entry point in `lfcbenchmark.metrics`.
The metric-registry extension in the accompanying LFCBenchmark checkout exposes
it automatically after installation. Manual registration is also available.
See [registration and timing](docs/lfcbench.md).

## Validation

The [HPDPS-8 validation record](validation/hpdps8-20261109/README.md) includes
56 passing tests, a pure C++ check, an isolated wheel installation, real SDRBench
crop checks, and the observed OpenMP timing results.

Run the validation on **HPDPS-8**, using a separate environment and the other CPU
socket while an existing benchmark is active. The script rejects other hosts.

```bash
python -m pip install -e '.[test]'
python -m pip install build
bash scripts/validate_hpdps8.sh
numactl --physcpubind=32-39 --membind=1 \
  python scripts/benchmark_kernels.py --shape 128 128 128 --threads 1 2 4 8
```

Set `LFCBENCH_CHECKOUT` to the integration checkout to include its tests. The
defaults bind to CPUs 32–39 and NUMA node 1; `LFCQUALITY_CPUS` and
`LFCQUALITY_NODE` can change them. Metric-accuracy unit tests use synthetic
references. A separate real-data integration check runs all seven metric groups
on ZFP reconstructions of leading 128³ NYX float32 and Miranda float64 crops:

```bash
python -m pip install zfpy
OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
  numactl --physcpubind=32-39 --membind=1 \
  python scripts/sdrbench_smoke.py --root /data0/lb/sdrbench \
  --output results/sdrbench-smoke
```

It checks the absolute bound and exact agreement between 1 and 8 OpenMP threads
on the same binary. It is a module integration test on declared crops, not a
full-field SDRBench compressor performance study.

NumPy, SciPy and scikit-image are independent **test** references only; no SciPy
or scikit-image code is used by the production metrics. Tests include 2D/3D SSIM,
non-wrapping/periodic ACF, odd/even multidimensional spectra, Parseval, constant
fields, affine transforms, large offsets, input rejection and thread consistency.

Our source is BSD-3-Clause. FFTW is an external dependency with its own GPL
license; this repository does not vendor or distribute FFTW binaries.
