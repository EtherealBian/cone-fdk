"""Publish two byte-identical GB10 binaries linked to the original residual arithmetic."""

import argparse
import json
import platform
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from build_residual import HERE, ROOT, FLAGS, RULES, sha
from spark_publication import fresh, publish, staging

FROZEN = "15a6def20cee4913fc2b9f2b040014a339afdf6be1c7ad00f50d66f22d0bb401"


def build_mapped(out):
    if (
        platform.machine() != "aarch64"
        or platform.system() != "Linux"
        or sha(ROOT / "fixtures/volume-residual-protocol-v1.json") != RULES
    ):
        raise ValueError("qualified platform and protocol required")
    out = fresh(out)
    src = HERE / "residuals/mapped_residual.cu"
    base = src.with_name("volume_residual.cu")
    h = sha(src)
    nvcc = "/usr/local/cuda/bin/nvcc"
    with staging(out, ".mapped-build-") as t:
        stage = Path(t) / "stage"
        stage.mkdir()
        hashes = []
        for i in range(2):
            with TemporaryDirectory(dir=t) as tmp:
                path = Path(tmp)
                exe = path / "mapped-residual"
                if sha(base) != FROZEN or sha(src) != h:
                    raise ValueError("source changed")
                subprocess.run([nvcc, *FLAGS, src, "-o", exe.name], cwd=tmp, check=True)
                if sha(base) != FROZEN or sha(src) != h:
                    raise ValueError("source changed")
                hashes.append(sha(exe))
                if not i:
                    shutil.copy2(exe, stage / exe.name)
        if len(set(hashes)) != 1 or sha(stage / "mapped-residual") != hashes[0]:
            raise ValueError("independent builds differ")
        doc = dict(
            format="CFDK-GB10-mapped-build-v1",
            protocol_sha256=RULES,
            kernel_source_sha256=FROZEN,
            transport_sha256=h,
            flags=FLAGS,
            independent_elf_sha256=hashes,
            builder_sha256=sha(__file__),
            compiler=subprocess.check_output([nvcc, "--version"], text=True).strip(),
        )
        (stage / "build.json").write_text(json.dumps(doc, indent=2) + "\n")
        publish(stage, out)
    return sha(out / "build.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    print(build_mapped(p.parse_args().out))
