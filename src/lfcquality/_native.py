"""Small ctypes layer; CDLL releases the GIL during every native calculation."""
from __future__ import annotations

import ctypes as ct
from functools import lru_cache
import hashlib
import json

import numpy as np

from .build import library_path, source_digest

P = ct.c_void_p
I = ct.c_int
N = ct.c_int64
D = ct.c_double


class Native:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path))
        signatures = {
            "lfq_abi_version": [], "lfq_openmp": [], "lfq_fftw": [], "lfq_fftw_threads": [],
            "lfq_team_size": [I], "lfq_stats": [P, P, I, N, I, P],
            "lfq_basic": [P, D, P],
            "lfq_acf": [P, P, I, P, I, I, N, I, P, I, P],
            "lfq_ssim": [P, P, I, P, I, I, I, D, P, I, P],
            "lfq_cross": [P, I, P, I, N, P, P, I, P],
            "lfq_spectral_fields": [P, P, I, P, I, P, P, I, I, I, I, P, P],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = signature, I
        self.lib.lfq_last_error.argtypes = []
        self.lib.lfq_last_error.restype = ct.c_char_p
        self.lib.lfq_fftw_version.argtypes = []
        self.lib.lfq_fftw_version.restype = ct.c_char_p
        if self.lib.lfq_abi_version() != 1:
            raise RuntimeError("incompatible LFCQuality native ABI; rebuild the library")
        self.info = {"abi_version": 1, "openmp": self.lib.lfq_openmp(),
                     "fftw": bool(self.lib.lfq_fftw()), "fftw_threads": bool(self.lib.lfq_fftw_threads()),
                     "fftw_version": self.lib.lfq_fftw_version().decode(),
                     "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        metadata = path.parent / "build.json"
        self.info["build"] = json.loads(metadata.read_text()) if metadata.is_file() else {"external_library": str(path)}
        if metadata.is_file() and self.info["build"]["source_sha256"] != source_digest():
            raise RuntimeError("native source/library mismatch; rebuild the library")

    def call(self, name, *args):
        values = [P(a.ctypes.data) if isinstance(a, np.ndarray) else a for a in args]
        if getattr(self.lib, name)(*values) != 0:
            raise ValueError(self.lib.lfq_last_error().decode("utf-8", errors="replace"))

    def statistics(self, x, y, threads):
        out = np.empty(18, dtype=np.float64)
        self.call("lfq_stats", x, y, x.dtype.itemsize, x.size, threads, out)
        return out


@lru_cache(maxsize=4)
def _load(path):
    return Native(path)


def native():
    return _load(library_path())
