import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lfcquality.cli import main
from lfcquality.io import load_field


def test_raw_and_npy_cli_provenance_and_no_overwrite(tmp_path):
    x = np.arange(729,dtype=np.float32).reshape(9,9,9)
    np.save(tmp_path/"x.npy",x)
    x.tofile(tmp_path/"x.raw")
    assert np.array_equal(load_field(tmp_path/"x.raw",dtype="float32",shape=[9,9,9]),x)
    out = tmp_path/"quality.json"
    args = ["compare",str(tmp_path/"x.npy"),str(tmp_path/"x.raw"),"--dtype","float32","--shape","9","9","9",
            "--threads","2","--metrics","psnr","pcc","--output",str(out)]
    assert main(args) == 0
    result = json.loads(out.read_text())
    assert len(result["inputs"]["original"]["sha256_file"]) == 64
    assert result["metrics"]["psnr"]["status"] == "positive_infinity"
    contents = out.read_bytes()
    assert main(args) == 2 and out.read_bytes() == contents
    with pytest.raises(ValueError,match="byte count"):
        load_field(tmp_path/"x.raw",dtype="float32",shape=[9,9,8])


def test_fields_manifest_relative_paths(tmp_path):
    x = np.arange(81,dtype=float).reshape(9,9)
    np.save(tmp_path/"x.npy",x)
    spec = {"grid_id":"grid","time_id":"t0","options":{"metrics":["pcc"],"threads":2},
            "fields":[{"name":name,"original":"x.npy","reconstructed":"x.npy"} for name in ("density","temperature")]}
    (tmp_path/"fields.json").write_text(json.dumps(spec))
    out = tmp_path/"report.json"
    assert main(["compare-fields",str(tmp_path/"fields.json"),"-o",str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["cross_variable_correlation"]["summary"]["max_absolute_delta"] == 0


def test_serial_no_fftw_build_has_explicit_capabilities(tmp_path):
    # A separate process avoids unloading/replacing a library used by other tests.
    program = """
from lfcquality import build_native, analyze
import numpy as np
build_native(openmp='off', fftw='off')
x=np.arange(100,dtype=float)
assert analyze(x,x,metrics=('pcc',))['metrics']['pcc']['value'] > .999
for options in ({'threads':2,'metrics':('pcc',)}, {'metrics':('spectral',)}):
    try:
        analyze(x,x,**options)
    except (ValueError,RuntimeError):
        pass
    else:
        raise AssertionError('missing capability was silently ignored')
"""
    env = {**os.environ,"LFCQUALITY_CACHE":str(tmp_path/"cache")}
    env.pop("LFCQUALITY_LIBRARY",None)
    subprocess.run([sys.executable,"-c",program],env=env,check=True,capture_output=True,text=True)
