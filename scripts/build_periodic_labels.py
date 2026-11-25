"""Build the optional periodic-label companion twice, without touching frozen CT binaries."""

import argparse
import ctypes as C
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def ptx(src, dest):
    lib = Path(sys.prefix) / "Lib/site-packages/nvidia/cuda_nvrtc/bin"
    with os.add_dll_directory(str(lib)):
        _builtins = C.CDLL(str(next(lib.glob("nvrtc-builtins64*.dll"))))
        dll = C.CDLL(str(next(p for p in lib.glob("nvrtc64_*.dll") if ".alt" not in p.name)))
        prog, size = C.c_void_p(), C.c_size_t()

        def ck(rc):
            if rc:
                raise RuntimeError(f"NVRTC status {rc}")

        ck(
            dll.nvrtcCreateProgram(
                C.byref(prog), src.read_bytes(), src.name.encode(), 0, None, None
            )
        )
        try:
            opts = (C.c_char_p * 2)(b"--gpu-architecture=compute_80", b"--std=c++14")
            rc = dll.nvrtcCompileProgram(prog, 2, opts)
            if rc:
                ck(dll.nvrtcGetProgramLogSize(prog, C.byref(size)))
                msg = C.create_string_buffer(size.value)
                ck(dll.nvrtcGetProgramLog(prog, msg))
                raise RuntimeError(msg.value.decode(errors="replace"))
            ck(dll.nvrtcGetPTXSize(prog, C.byref(size)))
            buf = C.create_string_buffer(size.value)
            ck(dll.nvrtcGetPTX(prog, buf))
            dest.write_bytes(buf.raw)
        finally:
            ck(dll.nvrtcDestroyProgram(C.byref(prog)))


def build(out, cxx):
    compiler = shutil.which(cxx)
    if not compiler:
        raise ValueError("Set --cxx to an x64 MinGW g++ executable")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ, PATH=str(Path(compiler).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    host = ROOT / "src/periodic_labels.cpp"
    device = ROOT / "src/periodic_labels.cu"
    exe, asm = out / "periodic-labels.exe", out / "periodic-labels.ptx"
    first = None
    for _ in range(2):
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c++14",
                "-static",
                "-Wl,--no-insert-timestamp",
                host,
                "-lshell32",
                "-o",
                exe,
            ],
            env=env,
            check=True,
        )
        ptx(device, asm)
        pair = {p.name: sha(p) for p in (exe, asm)}
        if first is not None and first != pair:
            raise RuntimeError("nonidentical second build")
        first = pair
    record = dict(
        format="CFDK-periodic-label-build-v1",
        identical_builds=2,
        compiler=subprocess.check_output(
            [compiler, "--version"], text=True, env=env
        ).splitlines()[0],
        nvrtc=version("nvidia-cuda-nvrtc-cu12"),
        products=first,
        sources={
            p.name: sha(p)
            for p in (
                host,
                device,
                host.with_name("files.hpp"),
                host.with_name("gpu.hpp"),
                Path(__file__),
            )
        },
    )
    (out / "build.json").write_bytes((json.dumps(record, indent=2) + "\n").encode())
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--cxx", default=os.getenv("CXX", "g++"))
    a = p.parse_args()
    build(a.out, a.cxx)
