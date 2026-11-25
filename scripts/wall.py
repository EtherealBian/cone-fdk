"""Read or measure a CFDKCYL1 atlas. Point records retain invalid status; never impute a wall."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from common import BUILD, dll_dirs

WALL_DTYPE = np.dtype(
    [
        ("inner", "<f8"),
        ("outer", "<f8"),
        ("thickness", "<f8"),
        ("status", "<u4"),
        ("reserved", "<u4"),
    ]
)
STATUS = ("measured", "empty", "clipped", "multiple", "ambiguous", "nonfinite", "undersampled")


def run_native_json(cmd):
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([*map(str, dll_dirs()), env.get("PATH", "")])
    run = subprocess.run(
        list(map(str, cmd)),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf8",
        errors="replace",
    )
    if run.returncode:
        raise RuntimeError(run.stderr.strip())
    return json.loads(run.stdout)


def measure(atlas, spec, out, level, *, budget=16, sector_cols=None, build=BUILD):
    cmd = [
        build / "cone-fdk.exe",
        "wall",
        "--ptx",
        build / "kernels.ptx",
        "--input",
        atlas,
        "--cylinder",
        spec,
        "--output",
        out,
        "--threshold",
        level,
        "--budget",
        budget,
    ]
    if sector_cols is not None:
        cmd += ["--sector-cols", sector_cols]
    return run_native_json(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--sector-cols", type=int)
    a = parser.parse_args()
    print(
        json.dumps(
            measure(
                a.atlas, a.spec, a.output, a.threshold, budget=a.budget, sector_cols=a.sector_cols
            ),
            indent=2,
        )
    )
