"""Build the optional bounded-row CUDA interface encoder twice, without changing the CT backend."""

import argparse
import json
import os
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

from build_periodic_labels import ROOT, ptx, sha


def build(out, cxx):
    compiler = shutil.which(cxx)
    if not compiler:
        raise ValueError("Set --cxx to an x64 MinGW g++ executable")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ, PATH=str(Path(compiler).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    host = ROOT / "src/periodic_interfaces.cpp"
    device = ROOT / "src/periodic_interfaces.cu"
    exe, asm = out / "periodic-interfaces.exe", out / "periodic-interfaces.ptx"
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
        format="CFDK-periodic-interface-build-v1",
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
                Path(__file__).with_name("build_periodic_labels.py"),
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
