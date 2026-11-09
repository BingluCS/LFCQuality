# Metric definitions

Let `x` be the original field, `y` the reconstructed field, `e=y-x`, and `N`
the number of samples. Axes and shape use C order, slowest to fastest. All input
values must be finite. The following conventions are part of output schema 1.

## Errors, PSNR and PCC

`RMSE = sqrt(sum(e²)/N)`, `mean_error = sum(e)/N`, and
`max_abs_error = max(abs(e))`.

The default range is `L=max(x)-min(x)`. An explicitly supplied positive
`data_range` replaces L for both PSNR and SSIM. Then:

```
NRMSE = RMSE / L
PSNR  = 20 log10(L / RMSE)
PCC   = sum((x-mean(x))*(y-mean(y))) /
        sqrt(sum((x-mean(x))²)*sum((y-mean(y))²))
```

PCC is also reported as `one_minus_pcc`. Separate scaling of centered x/y
avoids square overflow and cancellation in `E[x²]-E[x]²`. RMSE uses a scaled
sum of squares. Inputs whose range or difference itself overflows float64
are rejected. This implementation is not an arbitrary-precision calculator;
floating-point underflow and rounding are still possible at extreme scales.

When L=0, PSNR/NRMSE/SSIM are undefined unless the caller supplies a positive
range. With positive L and RMSE=0, PSNR is positive infinity. JSON encodes it
as `null` with `status: positive_infinity`. PCC is undefined if either variance
is zero, even when two constant arrays are identical.

## Local SSIM

At each valid window, use uniform weights and population (division by window
sample count) moments:

```
SSIM_w = ((2 μx μy + C1) / (μx² + μy² + C1)) *
         ((2 cov(x,y) + C2) / (var(x) + var(y) + C2))
C1 = (0.01 L)²
C2 = (0.03 L)²
```

The reported score is the arithmetic mean over window scores. Windows have
`ssim_window` elements per spatial axis, starting at coordinate 0 and then
advancing by `ssim_stride`; only fully contained windows are included. The
default is window 7 and stride 4. Remainder regions after the last valid window
start are not given a special additional window. Stride 1 evaluates every valid
window, and uniform-window skimage SSIM with population covariance is used as
an independent test reference. Three-dimensional inputs use volumes, not slices.

This is a declared uniform-window variant of the SSIM formula, not the original
paper's specific Gaussian-window image experiment. It is neither MS-SSIM nor
a domain-specific guarantee of physical feature preservation. A window larger
than any axis yields an explicit undefined status. Negative scores are retained.
The implementation uses centered two-pass window moments and scales all terms
before forming products; it avoids five full-field temporary moment arrays.

## Error and field autocorrelation

For a field `u`, an axis `a`, and lag `l`, define:

```
ACF(u,a,l) = sum_{i in valid pairs} (u[i]-mean(u))*(u[i+l*axis_a]-mean(u)) /
             sum_{i in full field} (u[i]-mean(u))²
```

The default `valid` mode excludes pairs outside the specified axis. Its
denominator is the **full-field** centered sum of squares; it does not divide
by the pair count or recenter each overlapping slab. This is the spatial-axis
extension of the NIST finite-sample ACF definition. It is not an unbiased
lag-covariance estimator. `periodic` mode wraps within that axis and includes N
pairs. No mode connects the end of a flattened row to the start of another row.

For error ACF set `u=e`. For field ACF compare `u=x` and `u=y` and report the
signed change `ACF(y)-ACF(x)` and its absolute value. Summary averages use all
valid requested axis/lag entries with equal weights; `valid_count` is reported.
Default lags are 1, 2, 4, 8, but 0 is allowed. Lags >= axis length are explicitly
undefined in both boundary modes. ACF of constant errors is undefined, including
exact reconstruction: an undefined EAC must not be interpreted as white noise.

## Spectral error

All spatial axes are transformed. Grid spacing must represent regularly sampled
coordinates with a common physical unit on every axis; for a nonuniform grid,
resample appropriately before calling this API. Unit spacing reports cycles per
grid cell. Different axis lengths and spacings are supported.

By default x and y are demeaned independently; `spectral_demean=False` keeps DC.
A rectangular window is used. Finite nonperiodic boundaries can therefore cause
spectral leakage; this is part of the declared convention. A shared amplitude
scale `S=max(max(abs(x)),max(abs(y)))`, or 1 if both are zero, keeps FFT power
calculations in range. For the transformed (optionally demeaned) fields:

```
Fx = DFT(x/S)/N
Fy = DFT(y/S)/N
Px = |Fx|²
Py = |Fy|²

power_relative_l1 = sum_k |Py[k]-Px[k]| / sum_k Px[k]
power_relative_l2 = sqrt(sum_k (Py[k]-Px[k])² / sum_k Px[k]²)
fourier_relative_l2 = sqrt(sum_k |Fy[k]-Fx[k]|² / sum_k |Fx[k]|²)
```

These sums represent the full frequency grid. Only a real FFT half-spectrum is
stored, with multiplicity 2 except at the last-axis zero-frequency plane and
even-length Nyquist plane. This distinction is tested against complete complex
FFTs, including odd-sized and anisotropic 2D/3D shapes.

The returned power is in normalized units: multiply by S² to recover physical
mean-square units where that product is representable. With demeaning, the sum
of the original spectrum is `var(x)/S²`, a Parseval check. Fourier relative L2 is
phase-sensitive and equals the corresponding relative spatial L2 error by
Parseval; it is included as a check, not as an independent spatial quality score.
Power errors can be zero for a pure spatial translation with altered phase.

Radial shells cover frequency magnitudes from 0 through the largest available
frequency magnitude, including corners beyond the smallest axis Nyquist radius.
They report **integrated power**, not mean power per mode or density per unit
frequency. Mode counts are returned so alternate averages can be computed.
No isotropy correction or incomplete-shell extrapolation is applied. Full-grid
power errors are calculated before shell aggregation, avoiding cancellation of
opposite signed errors within a shell. Empty and zero-reference shells have
explicit statuses; no arbitrary epsilon is inserted into a relative error.

## Cross-variable correlation change

For aligned variables x1,...,xm and their reconstructions y1,...,ym, compute:

```
R_original[i,j]      = PCC(xi,xj)
R_reconstructed[i,j] = PCC(yi,yj)
delta[i,j]           = R_reconstructed[i,j] - R_original[i,j]
```

Matrices include named rows/columns. Undefined entries involving a constant
variable are null. Summary mean/max absolute change considers only valid `i<j`
entries, and reports both valid and total pair counts. Different fields may be
f32/f64, but each field's original and reconstruction must have matching dtype.
The caller must establish common grid, time, ordering and mask. Comparing
unrelated same-shaped snapshots is a scientifically different question.

## References and independence

- [QCAT documentation](https://github.com/szcompressor/qcat/blob/15b8ee2c2aaeba39705e221b2a5e019e5d8a85db/README.md):
  interface and analysis-scope reference, inspected 2026-11-09. No implementation
  source was used, copied, translated, vendored or linked.
- [Wang et al., SSIM research page and 2004 paper](https://ece.uwaterloo.ca/~z70wang/research/ssim/):
  structural similarity definition. Our window conventions are explicit above.
- [NIST autocorrelation](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm):
  finite-sample centered ACF normalization, extended here along spatial axes.
- [FFTW real-data transforms](https://www.fftw.org/fftw3_doc/Real_002ddata-DFTs.html)
  and [thread safety](https://www.fftw.org/fftw3_doc/Thread-safety.html): transform
  layout, normalization, and serialized planner operations.

LFCQuality does not reimplement an FFT algorithm. FFTW is the externally linked
transform backend. All metric accumulation and reductions are written here.
