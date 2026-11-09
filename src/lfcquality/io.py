"""Explicit C-order binary/NumPy field loading and file provenance."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


def load_field(path, *, dtype=None, shape=None):
    path = Path(path).expanduser().resolve(strict=True)
    if shape is not None:
        if not shape or any(isinstance(n,bool) or not isinstance(n,int) or n<=0 for n in shape):
            raise ValueError("shape must contain positive integers in slowest-to-fastest C order")
        shape = tuple(shape)
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if shape is not None and array.shape != shape:
            raise ValueError("npy shape differs from the supplied shape")
        if dtype is not None and array.dtype != np.dtype(dtype):
            raise ValueError("npy dtype differs from the supplied dtype")
    else:
        if dtype is None or shape is None:
            raise ValueError("raw files require explicit --dtype and --shape (slowest to fastest)")
        dt = np.dtype(dtype)
        if dt not in (np.dtype("float32"),np.dtype("float64")) or not dt.isnative:
            raise ValueError("raw dtype must be native-endian float32 or float64")
        expected = math.prod(shape)*dt.itemsize
        if path.stat().st_size != expected:
            raise ValueError(f"raw byte count mismatch: expected {expected}, got {path.stat().st_size}")
        array = np.memmap(path, dtype=dt, mode="r", shape=shape, order="C")
    return array


def file_info(path):
    path = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2),b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256_file": digest.hexdigest()}
