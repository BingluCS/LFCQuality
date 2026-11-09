from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__, analyze, analyze_fields, build_native
from .analysis import METRICS
from .io import file_info, load_field


def parser():
    p = argparse.ArgumentParser(prog="lfcquality", description="C++/OpenMP quality analysis of scientific fields")
    p.add_argument("--version", action="version", version=__version__)
    commands = p.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="explicitly compile the native C++17 library")
    build.add_argument("--openmp", choices=("required","off"), default="required")
    build.add_argument("--fftw", choices=("required","off"), default="required")
    build.add_argument("--compiler", help="C++ compiler command; otherwise CXX or c++")
    commands.add_parser("info", help="report actual native build capabilities")
    compare = commands.add_parser("compare", help="compare one original/reconstructed field pair")
    compare.add_argument("original", type=Path)
    compare.add_argument("reconstructed", type=Path)
    compare.add_argument("--dtype", help="raw dtype, e.g. float32 or float64")
    compare.add_argument("--shape", type=int, nargs="+", help="C order: slowest to fastest, e.g. z y x")
    compare.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)
    compare.add_argument("--threads", type=int, default=1, help="OpenMP threads for analysis kernels")
    compare.add_argument("--fft-threads", type=int, default=1, help="FFTW worker threads")
    compare.add_argument("--lags", nargs="+", type=int, default=[1,2,4,8])
    compare.add_argument("--boundary", choices=("valid","periodic"), default="valid", help="ACF boundaries")
    compare.add_argument("--data-range", type=float, help="PSNR/SSIM reference range; default original max-min")
    compare.add_argument("--ssim-window", type=int, default=7)
    compare.add_argument("--ssim-stride", type=int, default=4)
    compare.add_argument("--spacing", type=float, nargs="+", help="grid spacing in a common physical unit, per axis")
    compare.add_argument("--spectral-bins", type=int, default=64)
    compare.add_argument("--keep-mean", action="store_true", help="retain the DC component in spectral analysis")
    compare.add_argument("--max-fft-bytes", type=int, help="reject FFT primary buffers above this budget; FFTW scratch is extra")
    compare.add_argument("--output", "-o", type=Path, help="new JSON file (never overwrite); default stdout")
    fields = commands.add_parser("compare-fields", help="compare aligned variables using a JSON manifest")
    fields.add_argument("manifest", type=Path)
    fields.add_argument("--threads", type=int, help="override analysis threads in manifest options")
    fields.add_argument("--output", "-o", type=Path)
    return p


def _fields(args):
    root = args.manifest.expanduser().resolve(strict=True)
    spec = json.loads(root.read_text())
    if not isinstance(spec,dict) or set(spec)-{"fields","options","grid_id","time_id"}:
        raise ValueError("manifest accepts fields, options, grid_id and time_id")
    if not isinstance(spec.get("fields"),list):
        raise ValueError("manifest fields must be a list")
    options = spec.get("options",{})
    if not isinstance(options,dict):
        raise ValueError("manifest options must be a mapping")
    if args.threads is not None:
        options["threads"] = args.threads
    original, reconstructed, provenance = {}, {}, {}
    for field in spec["fields"]:
        if not isinstance(field,dict) or set(field)-{"name","original","reconstructed","shape","dtype"}:
            raise ValueError("field accepts name, original, reconstructed, shape and dtype")
        name = field.get("name")
        if not isinstance(name,str) or not name or name in original:
            raise ValueError("field names must be nonempty and unique")
        a,b = (root.parent/field[key] for key in ("original","reconstructed"))
        original[name] = load_field(a,dtype=field.get("dtype"),shape=field.get("shape"))
        reconstructed[name] = load_field(b,dtype=field.get("dtype"),shape=field.get("shape"))
        provenance[name] = {"original": file_info(a), "reconstructed": file_info(b)}
    result = analyze_fields(original,reconstructed,**options)
    result["inputs"] = provenance
    result["grid_id"], result["time_id"] = spec.get("grid_id"), spec.get("time_id")
    result["manifest"] = file_info(root)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        output = getattr(args,"output",None)
        if output is not None and output.exists():
            raise FileExistsError(f"output already exists: {output}")
        if args.command == "build":
            result = build_native(openmp=args.openmp,fftw=args.fftw,compiler=args.compiler)
        elif args.command == "info":
            from ._native import native
            result = native().info
        elif args.command == "compare-fields":
            result = _fields(args)
        else:
            original = load_field(args.original,dtype=args.dtype,shape=args.shape)
            reconstructed = load_field(args.reconstructed,dtype=args.dtype,shape=args.shape)
            result = analyze(original,reconstructed,metrics=args.metrics,threads=args.threads,fft_threads=args.fft_threads,
                             lags=args.lags,boundary=args.boundary,data_range=args.data_range,ssim_window=args.ssim_window,
                             ssim_stride=args.ssim_stride,spacing=args.spacing,spectral_bins=args.spectral_bins,
                             spectral_demean=not args.keep_mean,max_fft_bytes=args.max_fft_bytes)
            result["inputs"] = {"original": file_info(args.original), "reconstructed": file_info(args.reconstructed)}
        text = json.dumps(result,indent=2,allow_nan=False)+"\n"
        if output is None:
            print(text,end="")
        else:
            output.parent.mkdir(parents=True,exist_ok=True)
            with output.open("x",encoding="utf-8") as stream:
                stream.write(text)
            print(f"Wrote {output}",file=sys.stderr)
        return 0
    except (Exception,KeyboardInterrupt) as exc:
        print(f"lfcquality: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2
