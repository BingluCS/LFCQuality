"""Generate deterministic co-located variables and run every metric."""
import argparse
import json
from pathlib import Path

import numpy as np

from lfcquality import analyze_fields

parser = argparse.ArgumentParser()
parser.add_argument("--output",type=Path,default=Path("results/demo"))
parser.add_argument("--threads",type=int,default=4)
args = parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
rng = np.random.default_rng(20261109)
grid = np.indices((24,28,32),dtype=float)
density = (3+np.sin(grid[0]/3)+.5*np.cos(grid[2]/5)).astype(np.float32)
temperature = (1.5*density+.2*np.sin(grid[1]/4)).astype(np.float32)
originals = {"density":density,"temperature":temperature}
restored = {key:(a+rng.normal(0,.01,a.shape)).astype(a.dtype) for key,a in originals.items()}
fields = []
for name in originals:
    np.save(args.output/f"{name}.npy",originals[name])
    np.save(args.output/f"{name}.reconstructed.npy",restored[name])
    fields.append({"name":name,"original":f"{name}.npy","reconstructed":f"{name}.reconstructed.npy"})
options = {"threads":args.threads,"fft_threads":1,"ssim_window":7,"ssim_stride":4}
manifest = {"grid_id":"synthetic-24x28x32","time_id":"t0","fields":fields,"options":options}
(args.output/"fields.json").write_text(json.dumps(manifest,indent=2)+"\n")
report = analyze_fields(originals,restored,**options)
(args.output/"quality.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(f"Wrote {args.output / 'quality.json'}")
