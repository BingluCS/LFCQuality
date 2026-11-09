# C and C++ API reference

The public C header is `<lfcquality/lfcquality.h>`. The C++17 convenience header
is `<lfcquality/lfcquality.hpp>`; it calls the same C ABI and C++/OpenMP kernels.
Neither API requires a Python interpreter or NumPy. The existing low-level
`kernels.h` ABI remains available for existing clients; new integrations should
normally use the typed public API below.

The implementation is in [`api.cpp`](../src/lfcquality/native/api.cpp), and the
numerical kernels are in [`kernels.cpp`](../src/lfcquality/native/kernels.cpp).
The available metric definitions and their normalizations are specified in
[metrics.md](metrics.md). C, C++ and Python all use those same kernels.

## Build, install, and link

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
cmake --install build --prefix "$PWD/install"
```

On HPDPS-8, add `-DCMAKE_PREFIX_PATH=/data0/lb/lfcquality/.native-deps` to the
configure command. With a system FFTW installation the extra prefix is unnecessary.
The installed headers are `install/include/lfcquality/lfcquality.h` and
`lfcquality.hpp`; the library is `install/lib/liblfcquality.so` on Linux and
`install/lib/liblfcquality.dylib` on macOS. The installed shared library must be
able to find its OpenMP and FFTW runtime dependencies. CMake preserves discovered
non-system dependency directories in the installed library's runtime path.

Link an ordinary C11 or C++17 program without linking Python:

```bash
gcc -std=c11 examples/c_api.c -I"$PWD/install/include" \
  -L"$PWD/install/lib" -Wl,-rpath,"$PWD/install/lib" -llfcquality -lm -o build/c_example_manual
g++ -std=c++17 examples/cpp_api.cpp -I"$PWD/install/include" \
  -L"$PWD/install/lib" -Wl,-rpath,"$PWD/install/lib" -llfcquality -o build/cpp_example_manual
```

For another CMake project:

```cmake
find_package(LFCQuality 0.2 CONFIG REQUIRED)
target_link_libraries(my_program PRIVATE LFCQuality::lfcquality)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix; a complete **C-only** CMake
consumer is in [`examples/cmake_consumer/`](../examples/cmake_consumer/).
The exported target can also be linked by a C++ target. For a source-tree
integration, enable C++ in the parent project and use `add_subdirectory()`;
the target alias is the same.

## Input fields

```c
typedef struct lfq_field {
    const void *data;
    size_t count;
    lfq_dtype dtype;       /* LFQ_FLOAT32 or LFQ_FLOAT64 */
    int ndim;              /* 1, 2 or 3 */
    int64_t shape[3];      /* first ndim entries, slowest to fastest */
} lfq_field;
```

For example, a 3D array stored as `float x[nz][ny][nx]` uses
`{x, nz*ny*nx, LFQ_FLOAT32, 3, {nz,ny,nx}}`. The dimensions are in C order;
the last dimension varies fastest. Buffers must be nonempty, naturally aligned,
contiguous, native endian and finite. `count` is the number of **elements**, not
bytes, and must equal the product of the dimensions. Original and reconstructed
fields must have the same shape and dtype. The API validates metadata and values,
but cannot detect a caller that provides an invalid pointer or falsely declares
the size of the allocation.

Field views do not own the input; keep the buffer valid and unchanged until the
analysis call returns. Input buffers are not modified. Results are self-contained
and do not retain pointers to the input arrays.

## C: a single field

`lfq_default_options()` returns an initialized options struct. Passing a null
options pointer to `lfq_analyze` uses the same defaults. On success,
`lfq_analyze` returns 0 and writes an owned `lfq_result *`; on failure it returns
-1, writes `NULL` to `*out`, and `lfq_error_message()` describes the error.
The error message is thread-local and must be copied before the next API call on
that thread if the caller needs to retain it. No C++ exception escapes the C API.

```c
lfq_options options = lfq_default_options();
options.threads = 4;
options.fft_threads = 2;
lfq_result *result = NULL;
if (lfq_analyze(&original, &reconstructed, &options, &result) != 0) {
    fprintf(stderr, "%s\n", lfq_error_message());
    return 1;
}
const lfq_summary *summary = lfq_result_summary(result);
size_t n_acf = 0, n_bins = 0;
const lfq_acf_value *acf = lfq_result_acf(result, &n_acf);
const lfq_spectral_bin *bins = lfq_result_spectral_bins(result, &n_bins);
const lfq_spectral_summary *spectrum = lfq_result_spectrum(result);
/* Read summary, acf[0..n_acf), spectrum and bins[0..n_bins). */
lfq_result_free(result);
```

The result object owns all accessor data. Accessor pointers are borrowed and
remain valid only until `lfq_result_free(result)`. Passing `NULL` to the free
function is safe. A caller must not call `free()` or `delete` on an accessor or
the result. Read-only access to one result from multiple threads is allowed if
the owner does not free it concurrently.

Basic error statistics, NRMSE, PSNR and PCC are always computed. The `metrics`
bit mask controls the additional groups; 0 requests only basic metrics.

| Option | Default | Meaning |
| --- | --- | --- |
| `metrics` | `LFQ_METRIC_ALL` | Bitwise OR of `LFQ_METRIC_SSIM`, `LFQ_METRIC_ACF`, `LFQ_METRIC_SPECTRUM` |
| `threads` | 1 | OpenMP analysis threads |
| `fft_threads` | 1 | FFTW worker threads |
| `data_range` | 0 | 0 selects original max-min; otherwise a positive finite reference range |
| `ssim_window`, `ssim_stride` | 7, 4 | Uniform window and start-step on every axis |
| `boundary` | `LFQ_BOUNDARY_VALID` | Valid or `LFQ_BOUNDARY_PERIODIC` ACF boundaries |
| `lags`, `lag_count` | `NULL`, 0 | Default lags 1, 2, 4, 8; otherwise a borrowed array of distinct nonnegative `int64_t` lags |
| `spectral_bins`, `spectral_demean` | 64, 1 | Number of radial bins; subtract each field's mean before the transform |
| `spacing` | `{1,1,1}` | Positive grid spacings, first `ndim` entries, in a common unit |
| `max_fft_bytes` | 0 | Optional limit for primary FFT buffers, not total process RSS |

Always initialize options with `lfq_default_options()`; its `struct_size` field
enables an explicit ABI compatibility check. The `lags` array, when supplied,
only needs to remain alive through the call. SSIM uses fully contained windows;
stride 4 samples window starts, whereas stride 1 evaluates every valid start.

The result exposes:

- `lfq_summary`: original range, RMSE, maximum/mean error, NRMSE, PSNR, PCC,
  `1-PCC`, SSIM, window count and individual status fields.
- `lfq_acf_value`: one entry per requested axis/lag, in axis-major order. It
  includes original, reconstructed and error ACF, signed field ACF change,
  number of valid pairs and independent statuses.
- `lfq_spectral_summary`: phase-insensitive power errors, phase-sensitive
  Fourier relative L2 error, the common amplitude scale and the estimated
  primary-buffer memory. `lfq_spectral_bin` contains shell boundaries, full-grid
  mode counts, original/reconstructed powers and relative change.

Undefined values are NaN with an explicit `lfq_status`. In particular,
`LFQ_STATUS_ZERO_VARIANCE` is returned for a constant field or error, and a lag
larger than an axis is `LFQ_STATUS_LAG_OUT_OF_RANGE`. A zero reference range gives
`LFQ_STATUS_ZERO_RANGE`; a lossless field with a positive reference range gives
`LFQ_STATUS_POSITIVE_INFINITY` for PSNR. `LFQ_STATUS_NOT_REQUESTED` marks a
disabled metric, and spectral bins distinguish empty bins, zero reference
power and numeric range limitations. A successful call can contain undefined
individual metrics; inspect the status rather than assuming every number is finite.

## C: cross-variable correlation changes

`lfq_correlation_matrix` accepts arrays of co-located original fields and their
reconstructions. Each array has `field_count` entries, and the caller supplies
space for `field_count * field_count` results. The output is a row-major `N×N`
matrix: entry `(i,j)` is at `matrix[i*N+j]`. Each entry contains original PCC,
reconstructed PCC, their signed difference and status flags. Mixed float32 and
float64 variables are supported; within a pair, original and reconstruction
must match. All variables must have the same shape, and the caller is responsible
for confirming the same physical grid, time and mask.

```c
lfq_field originals[2] = {density, temperature};
lfq_field reconstructions[2] = {density_hat, temperature_hat};
lfq_correlation matrix[4];
if (lfq_correlation_matrix(originals, reconstructions, 2, 4, matrix, 4) != 0) {
    fprintf(stderr, "%s\n", lfq_error_message());
    return 1;
}
printf("PCC change = %g\n", matrix[1].delta); /* row 0, column 1 */
```

Each variable is scanned once to establish its moments; the pairwise PCC
calculations then use those moments. If a call fails, output contents should
not be used. Constant variables yield undefined matrix entries without making
the whole call fail.

## C++ convenience layer

`lfcquality::field(ptr, count, {dims...})` infers float32 or float64 and returns
a non-owning `lfq_field` view. `lfcquality::default_options()` returns the same
options as the C function. `lfcquality::analyze(x, y, options)` returns a
move-only `lfcquality::Result`; its destructor releases the C allocation.

- `result.summary()` and `result.spectrum()` are references valid while the
  `Result` object is alive and not moved from.
- `result.autocorrelation()` and `result.spectral_bins()` return independent
  `std::vector` copies.
- `lfcquality::correlation_matrix(original_fields, reconstructed_fields, threads)`
  returns an independent row-major `std::vector<lfq_correlation>`.

The C++ convenience layer throws `std::runtime_error` for C API failures and
`std::invalid_argument` for invalid wrapper arguments. The full runnable C and
C++ examples are [`examples/c_api.c`](../examples/c_api.c) and
[`examples/cpp_api.cpp`](../examples/cpp_api.cpp).

## Capabilities and compatibility

`lfq_c_api_version()` returns `LFQ_C_API_VERSION`; check it when a program loads
a library dynamically. `lfq_openmp_version()` returns 0 when OpenMP is absent,
and `lfq_has_fftw()`, `lfq_has_threaded_fftw()` and
`lfq_fftw_library_version()` report spectral backend capabilities.
`lfq_observed_threads(n)` probes the runtime team size, which can be smaller
than requested due to runtime or OS limits.

The default options request all groups. If FFTW was disabled with
`-DLFCQUALITY_FFTW=OFF`, unset `LFQ_METRIC_SPECTRUM` before analysis. If OpenMP
was disabled with `-DLFCQUALITY_OPENMP=OFF`, use `threads=1`. Unsupported
requests fail explicitly. FFTW planning is serialized within this library;
other independent users of FFTW in the same process must coordinate planning
according to [FFTW's thread-safety rules](https://www.fftw.org/fftw3_doc/Thread-safety.html).
