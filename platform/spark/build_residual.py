"""Build the additional residual auditor twice, without changing any frozen FDK products."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from spark_publication import fresh, publish, staging

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RULES = "e4ecf24baa6d6bee9146208a4b8cc587560de1839bfd4120d80212fbd6a8d272"
FLAGS = [
    "-O3",
    "-std=c++17",
    "-arch=sm_121",
    "--fmad=false",
    "--ftz=false",
    "--frandom-seed=3997",
    "-Xlinker=--strip-all",
]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(out):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("native Spark build only")
    if sha(ROOT / "fixtures/volume-residual-protocol-v1.json") != RULES:
        raise ValueError("changed protocol")
    src = HERE / "residuals/volume_residual.cu"
    source_pin = sha(src)
    nvcc = "/usr/local/cuda/bin/nvcc"
    out = fresh(out)
    with staging(out, ".residual-build-") as tmp:
        stage = Path(tmp) / "accepted"
        stage.mkdir()
        builds = []
        for k in range(2):
            with tempfile.TemporaryDirectory(dir=tmp, prefix=f"build-{k}-") as t:
                work = Path(t)
                exe = work / "volume-residual"
                if sha(src) != source_pin:
                    raise ValueError("source changed before build")
                subprocess.run([nvcc, *FLAGS, src, "-o", exe.name], cwd=work, check=True)
                if sha(src) != source_pin:
                    raise ValueError("source changed during build")
                builds.append(sha(exe))
                if k == 0:
                    shutil.copy2(exe, stage / exe.name)
        if builds[0] != builds[1] or sha(stage / "volume-residual") != builds[0]:
            raise ValueError("two independent ELF products differ")
        result = dict(
            format="CFDK-residual-independent-build-v1",
            protocol_sha256=RULES,
            source_sha256=source_pin,
            build_script_sha256=sha(__file__),
            flags=FLAGS,
            compiler=subprocess.check_output([nvcc, "--version"], text=True).strip(),
            machine=platform.machine(),
            independent_elf_sha256=builds,
        )
        (stage / "build.json").write_text(json.dumps(result, indent=2) + "\n")
        publish(stage, out)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.out)))
