"""Report observed analysis-group timings; no compressor timings or input I/O."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time
from datetime import datetime, timezone

import numpy as np
from lfcquality import analyze

parser = argparse.ArgumentParser()
parser.add_argument("--shape",nargs="+",type=int,default=[128,128,128])
parser.add_argument("--threads",nargs="+",type=int,default=[1,2,4,8])
parser.add_argument("--repeats",type=int,default=3)
parser.add_argument("--output",type=Path)
args = parser.parse_args()
if args.repeats<1 or min(args.shape)<7 or min(args.threads)<1 or 1 not in args.threads:
    parser.error("shape axes >= 7, repeats/threads >= 1, and threads must include 1")
if args.output and args.output.exists():
    parser.error("output already exists")
rng = np.random.default_rng(31415)
x = rng.normal(size=args.shape).astype(np.float32)
y = (x+rng.normal(0,.001,x.shape)).astype(np.float32)
groups = {"basic":("psnr","pcc"),"acf":("error_autocorrelation","autocorrelation"),
          "ssim":("ssim",),"spectral":("spectral",)}
cases = [(name,t) for name in groups for t in args.threads]
random.Random(20261109).shuffle(cases)
measurements = {}
for name,t in cases:
    opts = {"metrics":groups[name],"threads":t,"fft_threads":1}
    analyze(x,y,**opts)
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = analyze(x,y,**opts)
        times.append(time.perf_counter()-start)
    measurements[name,t] = {"seconds":times,"median_seconds":statistics.median(times),
                             "team_probe":result["backend"]["omp_team_size_probe"]}
rows = []
for name in groups:
    baseline = measurements[name,1]["median_seconds"]
    for t in args.threads:
        row = {"group":name,"threads":t,**measurements[name,t]}
        row["speedup_vs_one"] = baseline/row["median_seconds"]
        rows.append(row)
        print(f"{name:10} threads={t:2} median={row['median_seconds']:.6f}s speedup={row['speedup_vs_one']:.2f}x")
report = {"shape":args.shape,"dtype":"float32","platform":platform.platform(),"host":platform.node(),
          "timestamp_utc":datetime.now(timezone.utc).isoformat(),
          "cpu_affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None,
          "thread_environment":{k:os.environ.get(k) for k in ("OMP_PROC_BIND","OMP_PLACES","OMP_WAIT_POLICY","OPENBLAS_NUM_THREADS")},
          "backend":result["backend"],
          "protocol":"one warmup per randomized group/thread case; API wall time including mandatory basic scans; input creation excluded; FFTW threads fixed at 1",
          "measurements":rows}
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
