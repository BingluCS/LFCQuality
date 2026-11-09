"""Explicit build of a Python-ABI-independent C++ library; no import-time compiler."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile


def source_digest():
    root = Path(__file__).parent / "native"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".cpp", ".h", ".hpp"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def cache_dir():
    base = Path(os.environ.get("LFCQUALITY_CACHE", Path.home() / ".cache" / "lfcquality"))
    return base / f"{platform.system()}-{platform.machine()}" / source_digest()[:20]


def library_path():
    override = os.environ.get("LFCQUALITY_LIBRARY")
    if override:
        return Path(override).expanduser().resolve(strict=True)
    selection = cache_dir() / "selected.json"
    if not selection.is_file():
        raise RuntimeError("native library is not built; run: lfcquality build --openmp required --fftw required")
    return Path(json.loads(selection.read_text())["library"]).resolve(strict=True)


def _prefix(name, env):
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser().resolve()
    if platform.system() == "Darwin" and shutil.which("brew"):
        proc = subprocess.run(["brew", "--prefix", name], text=True, capture_output=True)
        if proc.returncode == 0:
            root = Path(proc.stdout.strip())
            if root.is_dir():
                return root
    return None


def build_native(*, openmp="required", fftw="required", compiler=None):
    """Build with C++17; modes are 'required' or 'off'. Never silently falls back.

    Linux: install a C++ compiler and libfftw3-dev. macOS: brew install libomp fftw.
    CXX, CPPFLAGS, CXXFLAGS, LDFLAGS, LFCQUALITY_OMP_PREFIX and
    LFCQUALITY_FFTW_PREFIX are supported. No shell is used to run the compiler.
    """
    if openmp not in ("required", "off") or fftw not in ("required", "off"):
        raise ValueError("openmp and fftw must be 'required' or 'off'")
    if platform.system() not in ("Darwin", "Linux"):
        raise RuntimeError("native builds currently support Linux and macOS")
    command = shlex.split(compiler or os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        raise RuntimeError("C++ compiler not found; set CXX to a C++17 compiler")
    version = subprocess.run(command + ["--version"], check=True, text=True, capture_output=True).stdout.strip()
    source_root = Path(__file__).parent / "native"
    sources = [source_root / "kernels.cpp", source_root / "api.cpp"]
    args = command + ["-std=c++17", "-O3", "-fPIC", "-shared", "-pthread"]
    args += shlex.split(os.environ.get("CPPFLAGS", "")) + shlex.split(os.environ.get("CXXFLAGS", ""))
    # Finite checks and reproducible reductions require IEEE arithmetic.
    args += ["-fno-fast-math", "-ffp-contract=off"]
    links = []
    if openmp == "required":
        root = _prefix("libomp", "LFCQUALITY_OMP_PREFIX")
        if platform.system() == "Darwin" and "clang" in version.lower():
            if root is None:
                raise RuntimeError("Apple Clang needs libomp: brew install libomp (or set LFCQUALITY_OMP_PREFIX)")
            args += ["-Xpreprocessor", "-fopenmp", f"-I{root / 'include'}"]
            links += [f"-L{root / 'lib'}", f"-Wl,-rpath,{root / 'lib'}", "-lomp"]
        else:
            args += ["-fopenmp"]
    if fftw == "required":
        args += ["-DLFQ_FFTW", "-DLFQ_FFTW_THREADS"]
        root = _prefix("fftw", "LFCQUALITY_FFTW_PREFIX")
        if root:
            args += [f"-I{root / 'include'}"]
            links += [f"-L{root / 'lib'}", f"-Wl,-rpath,{root / 'lib'}"]
        links += ["-lfftw3_threads", "-lfftw3", "-lm"]
    links += shlex.split(os.environ.get("LDFLAGS", ""))
    specification = {"source_sha256": source_digest(), "compiler_version": version,
                     "compile_flags": args, "link_flags": links, "openmp": openmp, "fftw": fftw}
    key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:16]
    dest = cache_dir() / key
    dest.mkdir(parents=True, exist_ok=True)
    library = dest / ("liblfcquality.dylib" if platform.system() == "Darwin" else "liblfcquality.so")
    if not library.is_file():
        with tempfile.TemporaryDirectory(dir=dest) as directory:
            temporary = Path(directory) / library.name
            invocation = args + [str(source) for source in sources] + ["-o", str(temporary)] + links
            proc = subprocess.run(invocation, text=True, capture_output=True)
            if proc.returncode:
                raise RuntimeError("native build failed:\n" + shlex.join(invocation) + "\n" + proc.stderr)
            probe = ctypes.CDLL(str(temporary))
            if probe.lfq_abi_version() != 1 or probe.lfq_c_api_version() != 1 or bool(probe.lfq_openmp()) != (openmp == "required") or bool(probe.lfq_fftw()) != (fftw == "required"):
                raise RuntimeError("compiled native library failed capability checks")
            os.replace(temporary, library)
    specification["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    specification["library"] = str(library)
    (dest / "build.json").write_text(json.dumps(specification, indent=2) + "\n")
    with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir(), delete=False) as stream:
        json.dump({"library": str(library)}, stream)
        temp = stream.name
    os.replace(temp, cache_dir() / "selected.json")
    return specification
