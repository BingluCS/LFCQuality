import json
import math

import numpy as np
import pytest
from skimage.metrics import structural_similarity

from lfcquality import analyze, analyze_fields


@pytest.mark.parametrize("dtype",[np.float32,np.float64])
def test_basic_against_numpy(dtype):
    rng = np.random.default_rng(1729)
    x = rng.normal(2,3,(13,17,19)).astype(dtype)
    y = (x+rng.normal(0,.02,x.shape)).astype(dtype)
    m = analyze(x,y,metrics=("psnr","pcc"),threads=4)["metrics"]
    a,b = x.astype(float),y.astype(float)
    rmse = np.sqrt(np.mean((b-a)**2))
    assert m["error"]["rmse"] == pytest.approx(rmse,rel=2e-14)
    assert m["psnr"]["nrmse"] == pytest.approx(rmse/np.ptp(a),rel=2e-14)
    assert m["psnr"]["value_db"] == pytest.approx(20*np.log10(np.ptp(a)/rmse),abs=1e-11)
    assert m["pcc"]["value"] == pytest.approx(np.corrcoef(a.ravel(),b.ravel())[0,1],abs=2e-14)


@pytest.mark.parametrize("shape",[(19,),(13,17),(11,13,15)])
@pytest.mark.parametrize("stride",[1,3])
def test_local_ssim_against_independent_library(shape,stride):
    rng = np.random.default_rng(21)
    x = rng.normal(size=shape)
    y = .93*x+rng.normal(0,.1,shape)
    result = analyze(x,y,metrics=("ssim",),ssim_window=5,ssim_stride=stride,threads=4)["metrics"]["ssim"]
    _, image = structural_similarity(x,y,data_range=np.ptp(x),win_size=5,gaussian_weights=False,use_sample_covariance=False,full=True)
    reference = image[(slice(2,-2,stride),)*len(shape)].mean()
    assert result["value"] == pytest.approx(reference,rel=2e-12,abs=2e-12)
    assert result["window_count"] == math.prod((n-5)//stride+1 for n in shape)


def reference_acf(a,axis,lag,periodic):
    centered = a-a.mean()
    if periodic:
        other = np.roll(centered,-lag,axis=axis)
        return np.sum(centered*other)/np.sum(centered**2)
    first,second = [slice(None)]*a.ndim,[slice(None)]*a.ndim
    first[axis] = slice(0,a.shape[axis]-lag)
    second[axis] = slice(lag,None)
    return np.sum(centered[tuple(first)]*centered[tuple(second)])/np.sum(centered**2)


@pytest.mark.parametrize("shape",[(9,),(7,11),(5,7,9)])
@pytest.mark.parametrize("boundary",["valid","periodic"])
def test_acf_matches_spatial_reference(shape,boundary):
    rng = np.random.default_rng(72)
    x = rng.normal(size=shape)
    y = x+rng.normal(0,.2,shape)
    m = analyze(x,y,metrics=("error_autocorrelation","autocorrelation"),lags=(0,1,2,99),boundary=boundary,threads=4)["metrics"]
    for row in m["autocorrelation"]["values"]:
        axis,lag = row["axis"],row["lag"]
        if lag == 99:
            assert row["status"] == "lag_exceeds_axis" and row["delta"] is None
            continue
        a = reference_acf(x,axis,lag,boundary=="periodic")
        b = reference_acf(y,axis,lag,boundary=="periodic")
        assert row["original"] == pytest.approx(a,abs=3e-14)
        assert row["reconstructed"] == pytest.approx(b,abs=3e-14)
        assert row["delta"] == pytest.approx(b-a,abs=3e-14)
    for row in m["error_autocorrelation"]["values"]:
        if row["lag"] != 99:
            assert row["value"] == pytest.approx(reference_acf(y-x,row["axis"],row["lag"],boundary=="periodic"),abs=3e-14)


def test_acf_does_not_join_flattened_rows():
    x = np.array([[0.,1.,2.],[20.,30.,40.]])
    m = analyze(x,x,metrics=("autocorrelation",),lags=(1,))["metrics"]["autocorrelation"]["values"][1]
    assert m["pairs"] == 4
    assert m["original"] == pytest.approx(reference_acf(x,1,1,False))
    assert abs(m["original"]-reference_acf(x.ravel(),0,1,False)) > .01


@pytest.mark.parametrize("shape",[(5,),(6,),(5,7),(4,6),(3,5,7),(4,5,8)])
@pytest.mark.parametrize("demean",[False,True])
def test_fftw_against_full_numpy_fft_and_parseval(shape,demean):
    rng = np.random.default_rng(23)
    x = rng.normal(1,2,shape)
    y = .9*x+rng.normal(0,.2,shape)
    spacing = tuple(1.+i*.5 for i in range(len(shape)))
    s = analyze(x,y,metrics=("spectral",),spectral_bins=9,spectral_demean=demean,spacing=spacing,threads=4,fft_threads=2)["metrics"]["spectral"]
    scale = s["amplitude_scale"]
    a = (x-x.mean() if demean else x)/scale
    b = (y-y.mean() if demean else y)/scale
    fx,fy = np.fft.fftn(a,norm="forward"),np.fft.fftn(b,norm="forward")
    px,py = np.abs(fx)**2,np.abs(fy)**2
    assert s["original_power"] == pytest.approx(np.mean(a*a),rel=3e-13)
    assert s["reconstructed_power"] == pytest.approx(np.mean(b*b),rel=3e-13)
    assert s["power_relative_l1"] == pytest.approx(np.abs(py-px).sum()/px.sum(),rel=3e-12)
    assert s["power_relative_l2"] == pytest.approx(np.linalg.norm(py-px)/np.linalg.norm(px),rel=3e-12)
    assert s["fourier_relative_l2"] == pytest.approx(np.linalg.norm(b-a)/np.linalg.norm(a),rel=3e-12)
    assert sum(row["mode_count"] for row in s["shells"]) == x.size
    axes = np.meshgrid(*(np.fft.fftfreq(n,d=d) for n,d in zip(shape,spacing)),indexing="ij")
    radius = np.sqrt(sum(axis*axis for axis in axes))
    bins = np.minimum(8,(radius/radius.max()*9).astype(int))
    for index,row in enumerate(s["shells"]):
        mask = bins==index
        assert row["mode_count"] == np.count_nonzero(mask)
        assert row["original_power"] == pytest.approx(px[mask].sum(),rel=3e-12,abs=1e-16)
        assert row["reconstructed_power"] == pytest.approx(py[mask].sum(),rel=3e-12,abs=1e-16)


def test_power_and_phase_errors_are_distinct():
    x = np.sin(2*np.pi*np.arange(64)/64*5)
    y = np.roll(x,7)
    s = analyze(x,y,metrics=("spectral",))["metrics"]["spectral"]
    assert s["power_relative_l1"] < 1e-14
    assert s["fourier_relative_l2"] > 1


def test_constants_zero_error_and_strict_json():
    x = np.ones((9,9,9))
    report = analyze(x,x,threads=4)
    m = report["metrics"]
    assert m["pcc"]["status"] == "undefined_zero_variance"
    assert m["psnr"]["status"] == "undefined_zero_range"
    assert m["ssim"]["status"] == "undefined_zero_range"
    assert m["spectral"]["status"] == "undefined_zero_original_power"
    assert all(r["value"] is None for r in m["error_autocorrelation"]["values"])
    json.dumps(report,allow_nan=False)
    explicit = analyze(x,x,data_range=1.)["metrics"]
    assert explicit["psnr"]["status"] == "positive_infinity"
    assert explicit["ssim"]["value"] == 1.
    a = np.arange(9**3,dtype=float).reshape(9,9,9)
    lossless = analyze(a,a)["metrics"]
    assert lossless["psnr"]["status"] == "positive_infinity"
    assert lossless["pcc"]["value"] == pytest.approx(1)
    assert lossless["spectral"]["power_relative_l1"] == 0


def test_large_offsets_and_large_finite_magnitudes():
    x = (1e12+np.arange(10000)*.25).reshape(100,100)
    y = -2*x+3
    result = analyze(x,y,metrics=("psnr","pcc"),threads=4)["metrics"]
    assert result["pcc"]["value"] == pytest.approx(-1,abs=1e-14)
    x = np.linspace(1e200,2e200,10000)
    y = x*1.001
    m = analyze(x,y,metrics=("psnr","pcc"))["metrics"]
    assert math.isfinite(m["error"]["rmse"]) and m["psnr"]["value_db"] > 40


def test_cross_variable_matrices_mixed_dtype_and_constants():
    rng = np.random.default_rng(9)
    a = rng.normal(size=(13,17)).astype(np.float32)
    b = (a.astype(float)*2+rng.normal(0,.2,a.shape))
    c = np.ones_like(a)
    original = {"a":a,"b":b,"constant":c}
    reconstructed = {"a":a+.03,"b":np.flip(b,0).copy(),"constant":c}
    result = analyze_fields(original,reconstructed,metrics=("pcc",),threads=4)
    matrix = result["cross_variable_correlation"]
    expected_a = np.corrcoef(a.ravel().astype(float),b.ravel())[0,1]
    expected_b = np.corrcoef(reconstructed["a"].ravel().astype(float),reconstructed["b"].ravel())[0,1]
    assert matrix["original"][0][1] == pytest.approx(expected_a,abs=2e-14)
    assert matrix["delta"][0][1] == pytest.approx(expected_b-expected_a,abs=2e-14)
    assert matrix["original"][2] == [None,None,None]
    assert matrix["summary"]["valid_count"] == 1
    assert matrix["summary"]["total_off_diagonal_pairs"] == 3
    json.dumps(result,allow_nan=False)


def test_fixed_binary_openmp_thread_consistency():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(37,41,43)).astype(np.float32)
    y = (x+rng.normal(0,.03,x.shape)).astype(np.float32)
    one = analyze(x,y,threads=1,ssim_stride=3)
    four = analyze(x,y,threads=4,ssim_stride=3)
    assert one["metrics"] == four["metrics"]
    assert four["backend"]["openmp"] > 0
    assert four["backend"]["omp_team_size_probe"] >= 1


def test_input_validation_and_memory_budget():
    x = np.arange(729,dtype=float).reshape(9,9,9)
    bad = x.copy(); bad.flat[2] = np.nan
    for a,b in [(x,bad),(x,x.astype(np.float32)),(x,x[:4]),(x.astype(np.int64),x),(x.astype(">f8"),x)]:
        with pytest.raises((ValueError,TypeError)):
            analyze(a,b,metrics=("pcc",))
    for options in ({"threads":0},{"metrics":"pcc"},{"lags":(-1,)},{"data_range":0},{"spacing":(1,2)},
                    {"ssim_stride":0},{"boundary":"flatten"},{"fft_threads":0}):
        with pytest.raises((ValueError,TypeError)):
            analyze(x,x,**options)
    with pytest.raises(MemoryError):
        analyze(x,x,metrics=("spectral",),max_fft_bytes=1)
    a = x[:,::2,:]
    assert analyze(a,a,metrics=("pcc",))["metrics"]["pcc"]["value"] == pytest.approx(1)
    with pytest.raises(ValueError,match="same grid"):
        analyze_fields({"a":x,"b":x[:1]},{"a":x,"b":x[:1]})


def test_noncontiguous_spacing_and_concurrent_fft_calls():
    from concurrent.futures import ThreadPoolExecutor
    x = np.random.default_rng(28).normal(size=(9,11,13))
    y = np.roll(x,1,axis=1)
    spacing = np.arange(1.,7.)[::2]
    opts = {"metrics":("spectral",),"threads":2,"fft_threads":2,"spacing":spacing}
    expected = analyze(x,y,**opts)["metrics"]
    assert expected == analyze(x,y,**{**opts,"spacing":spacing.copy()})["metrics"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda _:analyze(x,y,**opts),range(4)))
    for report in reports:
        assert report["metrics"]["spectral"]["power_relative_l1"] == pytest.approx(expected["spectral"]["power_relative_l1"],abs=1e-14)
