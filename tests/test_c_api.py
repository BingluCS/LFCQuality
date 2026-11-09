"""Cross-check the public C ABI against the independently tested Python API."""
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
import pytest

from lfcquality import analyze, analyze_fields
from lfcquality._native import native


class Field(ct.Structure):
    _fields_ = [("data",ct.c_void_p),("count",ct.c_size_t),("dtype",ct.c_int),("ndim",ct.c_int),
                ("shape",ct.c_int64*3)]


class Options(ct.Structure):
    _fields_ = [("struct_size",ct.c_size_t),("metrics",ct.c_uint32),("threads",ct.c_int),("fft_threads",ct.c_int),
                ("data_range",ct.c_double),("ssim_window",ct.c_int),("ssim_stride",ct.c_int),("boundary",ct.c_int),
                ("lags",ct.POINTER(ct.c_int64)),("lag_count",ct.c_size_t),("spectral_bins",ct.c_int),
                ("spectral_demean",ct.c_int),("spacing",ct.c_double*3),("max_fft_bytes",ct.c_uint64)]


class Summary(ct.Structure):
    _fields_ = [("elements",ct.c_size_t)] + [(name,ct.c_double) for name in (
        "original_min","original_max","data_range","max_abs_error","mean_error","rmse","nrmse","psnr_db",
        "pcc","one_minus_pcc","ssim")] + [("ssim_windows",ct.c_uint64)] + [(name,ct.c_int) for name in (
        "nrmse_status","psnr_status","pcc_status","ssim_status")]


class Acf(ct.Structure):
    _fields_ = [("axis",ct.c_int),("lag",ct.c_int64),("pairs",ct.c_uint64)] + [(name,ct.c_double) for name in (
        "original","reconstructed","error","delta")] + [(name,ct.c_int) for name in (
        "original_status","reconstructed_status","error_status")]


class Spectrum(ct.Structure):
    _fields_ = [(name,ct.c_double) for name in ("amplitude_scale","original_power","reconstructed_power",
        "absolute_power_l1","power_relative_l1","power_relative_l2","fourier_relative_l2")] + [
        ("primary_buffer_bytes",ct.c_uint64),("status",ct.c_int)]


class Bin(ct.Structure):
    _fields_ = [("lower_frequency",ct.c_double),("upper_frequency",ct.c_double),("mode_count",ct.c_uint64),
                ("original_power",ct.c_double),("reconstructed_power",ct.c_double),("relative_delta",ct.c_double),
                ("status",ct.c_int)]


class Correlation(ct.Structure):
    _fields_ = [("original_pcc",ct.c_double),("reconstructed_pcc",ct.c_double),("delta",ct.c_double),
                ("original_status",ct.c_int),("reconstructed_status",ct.c_int)]


@pytest.fixture(scope="module")
def capi():
    lib = native().lib
    signatures = {
        "lfq_c_api_version": ([],ct.c_int),
        "lfq_default_options": ([],Options),
        "lfq_error_message": ([],ct.c_char_p),
        "lfq_analyze": ([ct.POINTER(Field),ct.POINTER(Field),ct.POINTER(Options),ct.POINTER(ct.c_void_p)],ct.c_int),
        "lfq_result_free": ([ct.c_void_p],None),
        "lfq_result_summary": ([ct.c_void_p],ct.POINTER(Summary)),
        "lfq_result_spectrum": ([ct.c_void_p],ct.POINTER(Spectrum)),
        "lfq_result_acf": ([ct.c_void_p,ct.POINTER(ct.c_size_t)],ct.POINTER(Acf)),
        "lfq_result_spectral_bins": ([ct.c_void_p,ct.POINTER(ct.c_size_t)],ct.POINTER(Bin)),
        "lfq_correlation_matrix": ([ct.POINTER(Field),ct.POINTER(Field),ct.c_size_t,ct.c_int,
                                     ct.POINTER(Correlation),ct.c_size_t],ct.c_int),
    }
    for name,(args,restype) in signatures.items():
        function = getattr(lib,name)
        function.argtypes, function.restype = args,restype
    assert lib.lfq_c_api_version() == 1
    assert lib.lfq_default_options().struct_size == ct.sizeof(Options)
    return lib


def field(array):
    assert array.flags.c_contiguous and array.dtype in (np.float32,np.float64)
    return Field(array.ctypes.data,array.size,array.dtype.itemsize,array.ndim,
                 (ct.c_int64*3)(*(list(array.shape)+[0]*(3-array.ndim))))


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
@pytest.mark.parametrize("shape",[(9,11),(9,10,13)])
@pytest.mark.parametrize("boundary",["valid","periodic"])
def test_c_api_matches_python_metrics(capi,dtype,shape,boundary):
    rng = np.random.default_rng(624)
    x = rng.normal(size=shape).astype(dtype)
    y = (x+rng.normal(0,.05,size=shape)).astype(dtype)
    a,b = field(x),field(y)
    options = capi.lfq_default_options()
    options.threads,options.fft_threads = 2,1
    options.ssim_window,options.ssim_stride,options.spectral_bins = 5,2,9
    options.boundary = 0 if boundary == "valid" else 1
    lags = (ct.c_int64*4)(0,1,2,20)
    options.lags,options.lag_count = lags,4
    spacing = tuple(1.+i*.5 for i in range(len(shape)))
    for i,value in enumerate(spacing): options.spacing[i] = value
    result = ct.c_void_p()
    assert capi.lfq_analyze(ct.byref(a),ct.byref(b),ct.byref(options),ct.byref(result)) == 0, capi.lfq_error_message()
    try:
        m = analyze(x,y,threads=2,fft_threads=1,ssim_window=5,ssim_stride=2,spectral_bins=9,
                    lags=(0,1,2,20),spacing=spacing,boundary=boundary)["metrics"]
        s = capi.lfq_result_summary(result).contents
        assert s.elements == x.size
        for name in ("rmse","max_abs_error","mean_error","original_min","original_max","data_range"):
            assert getattr(s,name) == pytest.approx(m["error"][name],abs=2e-14,rel=2e-14)
        assert s.psnr_db == pytest.approx(m["psnr"]["value_db"],abs=2e-13)
        assert s.nrmse == pytest.approx(m["psnr"]["nrmse"],abs=2e-14)
        assert s.pcc == pytest.approx(m["pcc"]["value"],abs=2e-14)
        assert s.ssim == pytest.approx(m["ssim"]["value"],abs=2e-14)
        assert s.ssim_windows == m["ssim"]["window_count"]
        assert (s.psnr_status,s.nrmse_status,s.pcc_status,s.ssim_status) == (0,0,0,0)
        count = ct.c_size_t()
        rows = capi.lfq_result_acf(result,ct.byref(count))
        ref = m["autocorrelation"]["values"]
        errors = m["error_autocorrelation"]["values"]
        assert count.value == len(ref)
        for i in range(count.value):
            row,other,error = rows[i],ref[i],errors[i]
            assert (row.axis,row.lag,row.pairs) == (other["axis"],other["lag"],other["pairs"])
            if other["delta"] is None:
                assert row.original_status == 4 and math.isnan(row.delta)
            else:
                assert row.original == pytest.approx(other["original"],abs=2e-14)
                assert row.reconstructed == pytest.approx(other["reconstructed"],abs=2e-14)
                assert row.delta == pytest.approx(other["delta"],abs=2e-14)
                assert row.error == pytest.approx(error["value"],abs=2e-14)
        spec = capi.lfq_result_spectrum(result).contents
        for name in ("amplitude_scale","original_power","reconstructed_power","absolute_power_l1",
                     "power_relative_l1","power_relative_l2","fourier_relative_l2"):
            assert getattr(spec,name) == pytest.approx(m["spectral"][name],rel=2e-13,abs=2e-14)
        bins = capi.lfq_result_spectral_bins(result,ct.byref(count))
        assert count.value == 9 and spec.status == 0
        assert spec.primary_buffer_bytes == m["spectral"]["primary_buffer_bytes"]
        for i in range(count.value):
            for name in ("lower_frequency","upper_frequency","original_power","reconstructed_power"):
                assert getattr(bins[i],name) == pytest.approx(m["spectral"]["shells"][i][name],rel=2e-13,abs=2e-14)
            assert bins[i].mode_count == m["spectral"]["shells"][i]["mode_count"]
    finally:
        capi.lfq_result_free(result)


def test_cross_variable_matrix_mixed_precision_and_undefined(capi):
    rng = np.random.default_rng(902)
    a = rng.normal(size=(9,11)).astype(np.float32)
    b = 1.5*a.astype(np.float64)+rng.normal(0,.2,a.shape)
    c = np.ones_like(a)
    ar = (a+rng.normal(0,.01,a.shape)).astype(a.dtype)
    br = np.flip(b,axis=0).copy()
    original = [a,b,c]
    reconstructed = [ar,br,c]
    first = (Field*3)(*[field(v) for v in original])
    second = (Field*3)(*[field(v) for v in reconstructed])
    matrix = (Correlation*9)()
    assert capi.lfq_correlation_matrix(first,second,3,2,matrix,9) == 0, capi.lfq_error_message()
    reference = analyze_fields(dict(zip("abc",original)),dict(zip("abc",reconstructed)),
                               metrics=("pcc",),threads=2)["cross_variable_correlation"]
    for i in range(3):
        for j in range(3):
            item = matrix[i*3+j]
            for name,key in (("original_pcc","original"),("reconstructed_pcc","reconstructed"),("delta","delta")):
                expected = reference[key][i][j]
                if expected is None: assert math.isnan(getattr(item,name))
                else: assert getattr(item,name) == pytest.approx(expected,abs=2e-14)
            if i == 2 or j == 2:
                assert item.original_status == 3 and item.reconstructed_status == 3
    assert capi.lfq_correlation_matrix(first,second,3,2,matrix,8) == -1
    assert b"too small" in capi.lfq_error_message()


def test_invalid_inputs_and_thread_local_errors(capi):
    x = np.arange(81,dtype=np.float32).reshape(9,9)
    a,b = field(x),field(x)
    options = capi.lfq_default_options()
    options.metrics = 0
    bad_count,bad_dtype = field(x),field(x)
    bad_count.count -= 1
    bad_dtype.dtype = 3
    def check(bad,word):
        for _ in range(30):
            result = ct.c_void_p(123)
            assert capi.lfq_analyze(ct.byref(bad),ct.byref(b),ct.byref(options),ct.byref(result)) == -1
            assert result.value is None and word in capi.lfq_error_message()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda args:check(*args),[(bad_count,b"count"),(bad_dtype,b"dtype")]))
    options.struct_size = 0
    result = ct.c_void_p()
    assert capi.lfq_analyze(ct.byref(a),ct.byref(b),ct.byref(options),ct.byref(result)) == -1
    assert b"incompatible ABI" in capi.lfq_error_message()


def test_result_does_not_borrow_input_buffers(capi):
    x = np.arange(81,dtype=np.float64).reshape(9,9)
    y = x+np.sin(x)*0.01
    a,b = field(x),field(y)
    options = capi.lfq_default_options()
    options.metrics = 3  # SSIM and ACF
    result = ct.c_void_p()
    assert capi.lfq_analyze(ct.byref(a),ct.byref(b),ct.byref(options),ct.byref(result)) == 0
    try:
        before = capi.lfq_result_summary(result).contents.rmse
        count = ct.c_size_t()
        rows = capi.lfq_result_acf(result,ct.byref(count))
        first_acf = rows[0].original
        x.fill(np.nan)
        y.fill(np.nan)
        assert capi.lfq_result_summary(result).contents.rmse == before
        assert rows[0].original == first_acf
        assert math.isnan(capi.lfq_result_spectrum(result).contents.original_power)
    finally:
        capi.lfq_result_free(result)
