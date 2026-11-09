"""HPDPS-8: real f32/f64 SDRBench crops, ZFP reconstructions, all seven metrics.

This is a quality-module integration check, not a compressor performance study.
zfpy is an optional test dependency, not a dependency of LFCQuality.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import numpy as np
import zfpy

from lfcquality import analyze_fields
from lfcquality.io import load_field

parser = argparse.ArgumentParser()
parser.add_argument("--root",type=Path,required=True)
parser.add_argument("--edge",type=int,default=128)
parser.add_argument("--threads",type=int,default=8)
parser.add_argument("--output",type=Path,required=True)
args = parser.parse_args()
if platform.node().split(".")[0] != "HPDPS-8":
    parser.error("this validation must run on HPDPS-8")
if args.edge<7:
    parser.error("edge must be >= 7")
args.output.mkdir(parents=True,exist_ok=False)
datasets = {
    "NYX":("float32",(512,512,512),("baryon_density.f32","temperature.f32")),
    "Miranda":("float64",(256,384,384),("density.d64","pressure.d64")),
}
summary = []
for dataset,(dtype,shape,names) in datasets.items():
    originals,restored,inputs = {},{},{}
    for name in names:
        source = args.root/dataset/name
        raw = load_field(source,dtype=dtype,shape=shape)
        original = np.ascontiguousarray(raw[tuple(slice(0,min(args.edge,n)) for n in shape)])
        bound = (float(np.max(original))-float(np.min(original)))*1e-3
        if bound<=0:
            raise ValueError(f"constant crop cannot use range-relative error: {source}")
        payload = zfpy.compress_numpy(original,tolerance=bound,write_header=True)
        reconstructed = zfpy.decompress_numpy(payload)
        originals[name],restored[name] = original,reconstructed
        inputs[name] = {"source":str(source),"full_shape":list(shape),"crop_shape":list(original.shape),
                        "crop_origin":[0]*len(shape),"alpha":1e-3,"absolute_bound":bound,
                        "crop_sha256":hashlib.sha256(memoryview(original).cast("B")).hexdigest(),
                        "reconstructed_sha256":hashlib.sha256(memoryview(reconstructed).cast("B")).hexdigest()}
    one = analyze_fields(originals,restored,threads=1,fft_threads=1)
    many = analyze_fields(originals,restored,threads=args.threads,fft_threads=1)
    for name in names:
        a,b = one["fields"][name]["metrics"],many["fields"][name]["metrics"]
        if a != b:
            raise AssertionError(f"OpenMP thread-count difference in {dataset}/{name}")
        if b["error"]["max_abs_error"] > inputs[name]["absolute_bound"]:
            raise AssertionError(f"ZFP tolerance violation: {dataset}/{name}")
    if one["cross_variable_correlation"] != many["cross_variable_correlation"]:
        raise AssertionError("cross-variable result differs between OpenMP thread counts")
    many["validation"] = {"host":platform.node(),"cpu_affinity":sorted(os.sched_getaffinity(0)),
                          "purpose":"real-data quality integration check; not codec performance",
                          "inputs":inputs,"codec":"zfpy", "codec_version":version("zfpy"),
                          "equal_metrics_at_threads":[1,args.threads]}
    with (args.output/f"{dataset}.json").open("x") as stream:
        json.dump(many,stream,indent=2,allow_nan=False)
        stream.write("\n")
    row = {"dataset":dataset,"dtype":dtype,"fields":len(names),"crop_shape":list(next(iter(originals.values())).shape),
           "metrics_equal":True,"cross_variable_delta":many["cross_variable_correlation"]["summary"]["max_absolute_delta"]}
    summary.append(row)
    print(json.dumps(row),flush=True)
with (args.output/"summary.json").open("x") as stream:
    json.dump(summary,stream,indent=2,allow_nan=False)
    stream.write("\n")
