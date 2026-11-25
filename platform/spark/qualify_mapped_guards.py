"""Analytic multi-tile, partial-band transport checks under four real CUDA diagnostics."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from mapped_transport import (
    HERE,
    MAPPED,
    ROOT,
    json_checked,
    mapped_account,
    need,
    sha,
    signature,
)
from qualify_residual import numpy_oracle
from residual_witness import agree
from spark_publication import fresh, publish, staging


def cases():
    # Exact powers of two, alternating signs, ties at tile endpoints, all-equal first plane.
    values = np.array([0.0, -0.0, 2**-20, -(2**-18), 2**-16, -(2**-14), 1.0, -1.0], dtype="<f4")
    y, x, z = 43, 37, 15
    i = np.arange(x * y * z)
    a = values[(i * 3) % len(values)].reshape(z, y * x).copy()
    b = values[(i // 7) % len(values)].reshape(a.shape).copy()
    a[0] = b[0]
    return [x, y, z], a, b


def qualification(out):
    out = fresh(out)
    root = ROOT.parent / "mapped-residual-build-001"
    br = json_checked(root / "build.json", MAPPED)
    exe = root / "mapped-residual"
    need(sha(exe) == br["independent_elf_sha256"][0], "binary changed")
    s, a, b = cases()
    oracle = [numpy_oracle(x.tobytes(), y.tobytes()) for x, y in zip(a, b, strict=True)]
    with staging(out, ".guard-") as t:
        stage = Path(t) / "accept"
        stage.mkdir()
        with TemporaryDirectory(dir=t) as tmp:
            wd = Path(tmp)
            c = wd / "candidate"
            r = wd / "reference"
            a.tofile(c)
            b.tofile(r)
            cmd = list(map(str, [exe, c, r, *s, 4, 1]))
            rows = []
            sig = None
            for tool in ["memcheck", "racecheck", "synccheck", "initcheck"]:
                flags = ["--leak-check", "full"] if tool == "memcheck" else []
                log = stage.resolve(strict=True) / (tool + ".log")
                p = subprocess.run(
                    [
                        "/usr/local/cuda/bin/compute-sanitizer",
                        "--tool",
                        tool,
                        "--error-exitcode",
                        "87",
                        "--log-file",
                        str(log),
                        *flags,
                        *cmd,
                    ],
                    capture_output=True,
                    timeout=180,
                    env=dict(os.environ, LC_ALL="C"),
                )
                need(
                    not p.stderr,
                    "unexpected sanitizer stderr: " + p.stderr.decode(errors="replace")[-500:],
                )
                text = log.read_text()
                clean = "ERROR SUMMARY: 0 errors" in text or (
                    tool == "racecheck" and "0 hazards displayed (0 errors, 0 warnings)" in text
                )
                need(
                    p.returncode == 0
                    and clean
                    and (tool != "memcheck" or "0 bytes leaked in 0 allocations" in text),
                    "sanitizer " + tool + ": " + text[-500:],
                )
                d = json.loads(p.stdout)
                mapped_account(d, s, 4, 1)
                for row, expected in zip(d["rows"], oracle, strict=True):
                    agree(row, expected)
                key = signature(d["rows"])
                need(sig in (None, key), "diagnostic changed result")
                sig = key
                rows.append(
                    dict(
                        tool=tool,
                        exit=p.returncode,
                        log_sha256=sha(log),
                        plane_witness_sha256=key,
                    )
                )
            doc = dict(
                format="CFDK-GB10-mapped-transport-sanitizers-v1",
                build=MAPPED,
                source_sha256=sha(__file__),
                shape_xyz=s,
                band=4,
                payload_sha256=[sha(c), sha(r)],
                independent_numpy_planes=len(oracle),
                records=rows,
                scope="Four tools on this analytic fixture. Not a claim about sanitizer coverage of a full physical volume.",
            )
            (stage / "qualification.json").write_text(json.dumps(doc, indent=2) + "\n")
        publish(stage, out)
    print(sha(out / "qualification.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    qualification(p.parse_args().out)
