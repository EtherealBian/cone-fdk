"""Run the source-build CUDA bore gauge. Input is a wall map, not a cylinder atlas."""

from common import BUILD
from wall import run_native_json


def gauge(wall, grid, out, limit, *, budget=16, build=BUILD):
    return run_native_json(
        [
            build / "cone-fdk.exe",
            "bore",
            "--ptx",
            build / "kernels.ptx",
            "--input",
            wall,
            "--cylinder",
            grid,
            "--output",
            out,
            "--max-residual-mm",
            limit,
            "--budget",
            budget,
        ]
    )
