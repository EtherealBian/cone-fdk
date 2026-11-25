"""Compare two already registered radial wall maps with the native CUDA inspector."""

import argparse
import json
from pathlib import Path

from common import BUILD
from wall import run_native_json


def inspect_loss(reference, reference_grid, part, grid, out, threshold, radius, *, budget=16):
    return run_native_json(
        [
            BUILD / "cone-fdk.exe",
            "loss",
            "--ptx",
            BUILD / "kernels.ptx",
            "--reference",
            reference,
            "--reference-grid",
            reference_grid,
            "--input",
            part,
            "--cylinder",
            grid,
            "--output",
            out,
            "--loss-mm",
            threshold,
            "--area-radius-mm",
            radius,
            "--budget",
            budget,
        ]
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "reference_grid", "part", "grid", "out"):
        p.add_argument(name, type=Path)
    p.add_argument("--loss-mm", type=float, required=True)
    p.add_argument("--area-radius-mm", type=float, required=True)
    p.add_argument("--budget", type=int, default=16)
    a = p.parse_args()
    print(
        json.dumps(
            inspect_loss(
                a.reference,
                a.reference_grid,
                a.part,
                a.grid,
                a.out,
                a.loss_mm,
                a.area_radius_mm,
                budget=a.budget,
            ),
            indent=2,
        )
    )
