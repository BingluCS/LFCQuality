"""Public array API. All data-sized numerical loops execute in C++/FFTW."""
from __future__ import annotations

from collections.abc import Mapping
import math
import numbers
import time

import numpy as np

from . import __version__
from ._native import native

METRICS = ("psnr", "ssim", "pcc", "error_autocorrelation", "autocorrelation", "spectral")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum or value > 2**31 - 1:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**31-1]")
    return int(value)


def _array(value, name):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype not in (np.dtype("float32"), np.dtype("float64")) or not value.dtype.isnative:
        raise ValueError(f"{name} must have native-endian float32 or float64 dtype")
    if not 1 <= value.ndim <= 3 or not value.size:
        raise ValueError(f"{name} must be a nonempty 1D, 2D or 3D field")
    # Read-only memmaps can pass through without copying. C++ checks finiteness.
    return np.require(value, requirements=["C", "A"])


def _value(value):
    return float(value) if math.isfinite(value) else None


def _summary(values):
    finite = [float(v) for v in values if v is not None]
    return {"valid_count": len(finite), "mean_absolute_delta": math.fsum(abs(v) for v in finite)/len(finite) if finite else None,
            "max_absolute_delta": max((abs(v) for v in finite), default=None)}


def _spectral(lib, x, y, s, shape, spacing, bins, demean, threads, fft_threads, max_fft_bytes):
    if not lib.info["fftw"]:
        raise RuntimeError("spectral analysis requires FFTW; run: lfcquality build --fftw required")
    if not lib.info["fftw_threads"] and fft_threads != 1:
        raise ValueError("native FFTW does not support multiple FFT threads")
    m = x.size // x.shape[-1] * (x.shape[-1] // 2 + 1)
    # Exact primary buffers; excludes FFTW internal planning/thread scratch.
    buffer_bytes = 8*x.size + 32*m + ((m-1)//65536+1)*(6+3*bins)*8
    if max_fft_bytes is not None and buffer_bytes > max_fft_bytes:
        raise MemoryError(f"FFT buffers need at least {buffer_bytes} bytes, exceeding max_fft_bytes={max_fft_bytes}")
    totals, shells = np.empty(10), np.empty(3*bins)
    lib.call("lfq_spectral_fields", x, y, x.dtype.itemsize, shape, x.ndim, s, spacing, bins, int(demean),
             threads, fft_threads, totals, shells)
    rmax = math.hypot(*(n//2/n/float(d) for n,d in zip(x.shape,spacing)))
    if not all(math.isfinite(v) for v in totals[:7]):
        raise FloatingPointError("non-finite spectral intermediate")
    rows = []
    for b in range(bins):
        px, py = float(shells[b]), float(shells[bins+b])
        count = int(shells[2*bins+b])
        rows.append({"lower_frequency": rmax*b/bins, "upper_frequency": rmax*(b+1)/bins,
                     "mode_count": count, "original_power": px, "reconstructed_power": py,
                     "relative_delta": _value((py-px)/px) if px > 0 else None,
                     "status": "empty_bin" if not count else "zero_original_power" if px == 0 else "ok"})
    return {"status": "ok" if totals[0] > 0 else "undefined_zero_original_power",
            "power_relative_l1": _value(totals[7]), "power_relative_l2": _value(totals[8]),
            "fourier_relative_l2": _value(totals[9]), "original_power": float(totals[0]),
            "reconstructed_power": float(totals[1]), "absolute_power_l1": float(totals[2]),
            "amplitude_scale": float(totals[6]), "demean": demean, "window": "rectangular",
            "fft_normalization": "forward: F=DFT(field/amplitude_scale)/N",
            "hermitian_weights": "2 except last-axis DC and even-length Nyquist planes",
            "radial_aggregation": "integrated power over all available modes in each shell",
            "spacing": spacing.tolist(), "frequency_unit": "cycles per spacing unit",
            "fft_threads": fft_threads, "primary_buffer_bytes": buffer_bytes, "shells": rows}


def _analyze(original, reconstructed, *, metrics=METRICS, threads=1, lags=(1,2,4,8), boundary="valid",
             data_range=None, ssim_window=7, ssim_stride=4, spacing=None, spectral_bins=64,
             spectral_demean=True, fft_threads=1, max_fft_bytes=None):
    started = time.perf_counter()
    x, y = _array(original, "original"), _array(reconstructed, "reconstructed")
    if x.shape != y.shape or x.dtype != y.dtype:
        raise ValueError("original and reconstructed must have identical shape and dtype")
    if isinstance(metrics, str):
        raise TypeError("metrics must be a sequence of names, not a string")
    metrics = tuple(metrics)
    if not metrics or any(m not in METRICS for m in metrics) or len(set(metrics)) != len(metrics):
        raise ValueError(f"metrics must be a nonempty selection from {METRICS}, without duplicates")
    threads, fft_threads = _integer(threads, "threads"), _integer(fft_threads, "fft_threads")
    ssim_window, ssim_stride = _integer(ssim_window, "ssim_window"), _integer(ssim_stride, "ssim_stride")
    spectral_bins = _integer(spectral_bins, "spectral_bins")
    if spectral_bins > 65536:
        raise ValueError("spectral_bins must be <= 65536")
    if not isinstance(spectral_demean, bool):
        raise ValueError("spectral_demean must be boolean")
    if boundary not in ("valid", "periodic"):
        raise ValueError("boundary must be 'valid' or 'periodic'")
    lags = tuple(_integer(lag, "lag", 0) for lag in lags)
    if not lags or len(set(lags)) != len(lags):
        raise ValueError("lags must be nonempty and unique")
    if data_range is not None:
        if isinstance(data_range, bool) or not isinstance(data_range, numbers.Real) or not math.isfinite(data_range) or data_range <= 0:
            raise ValueError("data_range must be finite and positive")
    if max_fft_bytes is not None:
        if isinstance(max_fft_bytes, bool) or not isinstance(max_fft_bytes, numbers.Integral) or max_fft_bytes <= 0:
            raise ValueError("max_fft_bytes must be a positive integer")
    spacing = np.ascontiguousarray((1.,)*x.ndim if spacing is None else spacing, dtype=np.float64)
    if spacing.shape != (x.ndim,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing must contain one finite positive value per axis")
    radius = math.hypot(*(n//2/n/float(d) for n,d in zip(x.shape,spacing)))
    if not math.isfinite(radius) or (radius == 0 and x.size > 1):
        raise ValueError("spacing is outside the supported float64 frequency range")
    lib = native()
    if threads > 1 and not lib.info["openmp"]:
        raise ValueError("threads > 1 requires OpenMP; rebuild with --openmp required")
    s = lib.statistics(x, y, threads)
    if s[13] == 0 and s[14] != 0:
        raise FloatingPointError("RMSE underflowed float64; rescale both inputs before analysis")
    shape = np.asarray(x.shape, dtype=np.int64)
    value_range = float(s[16]-s[0]) if data_range is None else float(data_range)
    basic = np.empty(4)
    lib.call("lfq_basic", s, value_range, basic)
    out = {"error": {"rmse": float(s[13]), "max_abs_error": float(s[14]), "mean_error": float(s[15]),
                     "original_min": float(s[0]), "original_max": float(s[16]),
                     "data_range": value_range, "range_source": "original_max_minus_min" if data_range is None else "explicit"}}
    if "psnr" in metrics:
        out["psnr"] = {"value_db": _value(basic[1]), "nrmse": _value(basic[0]),
                       "nrmse_status": "undefined_zero_range" if value_range == 0 else "positive_infinity" if math.isinf(basic[0]) else "ok",
                       "status": "undefined_zero_range" if value_range == 0 else "positive_infinity" if s[13] == 0 else "ok"}
    if "pcc" in metrics:
        out["pcc"] = {"value": _value(basic[2]), "one_minus_pcc": _value(basic[3]),
                      "status": "ok" if math.isfinite(basic[2]) else "undefined_zero_variance"}
    if "ssim" in metrics:
        result = {"value": None, "window": [ssim_window]*x.ndim, "stride": [ssim_stride]*x.ndim,
                  "weights": "uniform", "covariance": "population", "boundary": "valid",
                  "data_range": value_range, "K1": 0.01, "K2": 0.03, "window_count": 0}
        if min(x.shape) < ssim_window:
            result["status"] = "undefined_window_exceeds_shape"
        elif value_range == 0:
            result["status"] = "undefined_zero_range"
        else:
            values = np.empty(2)
            lib.call("lfq_ssim", x, y, x.dtype.itemsize, shape, x.ndim, ssim_window, ssim_stride, value_range, s, threads, values)
            if not math.isfinite(values[0]):
                raise FloatingPointError("non-finite SSIM intermediate")
            result.update(value=float(values[0]), window_count=int(values[1]), status="ok")
        out["ssim"] = result
    if "error_autocorrelation" in metrics or "autocorrelation" in metrics:
        errors, fields, deltas = [], [], []
        for axis, length in enumerate(x.shape):
            error_rows, field_rows = [], []
            for lag in lags:
                values = np.full(3, np.nan)
                pairs = 0 if lag >= length else x.size if boundary == "periodic" else (x.size//length)*(length-lag)
                if pairs:
                    lib.call("lfq_acf", x, y, x.dtype.itemsize, shape, x.ndim, axis, lag, int(boundary=="periodic"), s, threads, values)
                a,b,e = [_value(v) for v in values]
                delta = b-a if a is not None and b is not None else None
                common = {"axis": axis, "lag": lag, "pairs": pairs}
                error_rows.append({**common, "value": e, "status": "lag_exceeds_axis" if not pairs else "undefined_zero_variance" if e is None else "ok"})
                field_rows.append({**common, "original": a, "reconstructed": b, "delta": delta,
                                   "absolute_delta": abs(delta) if delta is not None else None,
                                   "status": "lag_exceeds_axis" if not pairs else "undefined_zero_variance" if delta is None else "ok"})
                deltas.append(delta)
            errors.extend(error_rows); fields.extend(field_rows)
        description = {"normalization": "global-centered lag-product sum / full-field centered-square sum", "boundary": boundary}
        if "error_autocorrelation" in metrics:
            out["error_autocorrelation"] = {**description, "error_definition": "reconstructed - original", "values": errors}
        if "autocorrelation" in metrics:
            out["autocorrelation"] = {**description, "values": fields, "summary": _summary(deltas)}
    if "spectral" in metrics:
        out["spectral"] = _spectral(lib,x,y,s,shape,spacing,spectral_bins,spectral_demean,threads,fft_threads,max_fft_bytes)
    result = {"schema_version": 1, "lfcquality_version": __version__, "shape": list(x.shape), "dtype": x.dtype.str,
              "backend": {**lib.info, "threads_requested": threads, "omp_team_size_probe": lib.lib.lfq_team_size(threads)},
              "metrics": out, "analysis_seconds": time.perf_counter()-started}
    return result, s, x, y


def analyze(original, reconstructed, **options):
    """Compare co-located finite float32/float64 fields (one, two or three axes).

    Defaults: all six single-field metric groups; threads=1; ACF lags=(1,2,4,8)
    with valid boundaries; volumetric uniform SSIM window=7, stride=4;
    FFTW spectrum with 64 radial bins, unit spacing, demeaning and fft_threads=1.
    Use ssim_stride=1 for every valid window. All parameters appear in the JSON.
    Undefined quantities are None plus a status; infinite PSNR is explicitly
    'positive_infinity'. The returned dictionary is strict-JSON serializable.
    """
    return _analyze(original, reconstructed, **options)[0]


def analyze_fields(originals, reconstructions, **options):
    """Per-field quality plus change of the inter-variable PCC matrix.

    Inputs are mappings of matching unique field names to arrays. Every field
    must refer to the same grid, time, coordinate ordering and mask. Shape is
    checked; the caller is responsible for the physical alignment. Fields may
    have different dtypes; each original/reconstruction pair must match.
    """
    if not isinstance(originals, Mapping) or not isinstance(reconstructions, Mapping):
        raise TypeError("originals and reconstructions must be mappings")
    if set(originals) != set(reconstructions) or len(originals) < 2:
        raise ValueError("at least two matching named fields are required")
    if any(not isinstance(k,str) or not k for k in originals):
        raise ValueError("field names must be nonempty strings")
    names = sorted(originals)
    shapes = {tuple(_array(originals[k], k).shape) for k in names}
    if len(shapes) != 1:
        raise ValueError("all fields must have the same grid shape")
    reports, moments, arrays = {}, {}, {}
    for name in names:
        report, stats, x, y = _analyze(originals[name], reconstructions[name], **options)
        reports[name], moments[name], arrays[name] = report, stats, (x,y)
    lib, threads = native(), _integer(options.get("threads",1), "threads")
    size = len(names)
    matrices = [[[None]*size for _ in range(size)] for _ in range(2)]
    for kind in (0,1):
        for i,name in enumerate(names):
            a = moments[name][[kind,3+kind,6+kind,9+kind]].copy()
            matrices[kind][i][i] = 1.0 if a[3] > 0 else None
            for j in range(i+1,size):
                other = names[j]
                b = moments[other][[kind,3+kind,6+kind,9+kind]].copy()
                x,y = arrays[name][kind], arrays[other][kind]
                value = np.empty(1)
                lib.call("lfq_cross",x,x.dtype.itemsize,y,y.dtype.itemsize,x.size,a,b,threads,value)
                matrices[kind][i][j] = matrices[kind][j][i] = _value(value[0])
    delta = [[None]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            a,b = matrices[0][i][j],matrices[1][i][j]
            if a is not None and b is not None:
                delta[i][j] = b-a
    summary = _summary([delta[i][j] for i in range(size) for j in range(i+1,size)])
    summary["total_off_diagonal_pairs"] = size*(size-1)//2
    return {"schema_version": 1, "lfcquality_version": __version__, "fields": reports,
            "cross_variable_correlation": {"field_names": names, "original": matrices[0], "reconstructed": matrices[1],
                                           "delta": delta, "summary": summary,
                                           "undefined_entry": "zero variance in either variable",
                                           "alignment": "caller-supplied co-located grid and time"}}
