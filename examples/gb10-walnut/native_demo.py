"""Run three native CUDA jobs on the bundled 90-view acquisition; only temporary outputs are created."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "platform/spark"))
from review_multigrid import read

GRIDS = ((225, 34.5, (94, 87, 96, 39, 53, 31)), (384, 28, (159, 147, 173, 47, 59, 27)))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(command):
    return json.loads(subprocess.check_output(list(map(str, command))))


def demo():
    data = ROOT / "fixtures/walnut-s8-b8"
    geo, archive = data / "corrected.geom", data / "corrected.npz"
    if (
        digest(geo) != "b7f1965a916f9ad97186f16db51b37514da8e68f33b43cb9248119874630688f"
        or digest(archive) != "2b647682771473f0ca06753b0c54c28f7a3c03540625008ec83fec3f4833833c"
    ):
        raise ValueError("unexpected 90-view fixture")
    base = ROOT / "build/spark"
    with tempfile.TemporaryDirectory(prefix="cone-fdk-demo-") as folder:
        tmp = Path(folder)
        scan = tmp / "views.f32"
        spec = tmp / "grids.txt"
        packed = tmp / "volume.mg"
        with np.load(archive, allow_pickle=False) as arrays:
            views = arrays["views"]
            if views.shape != (90, 296, 280) or views.dtype.str != "<f4":
                raise ValueError("unexpected fixture layout")
            views.tofile(scan)
        common = [
            "--geom",
            geo,
            "--input",
            scan,
            "--ptx",
            base / "kernels.ptx",
            "--budget",
            "256",
            "--batch",
            "16",
        ]
        rows = [",".join(map(str, box)) for _, _, box in GRIDS]
        spec.write_text(
            "CFDK-GRIDS-1\n"
            + "".join(f"{n} {fov} {row}\n" for (n, fov, _), row in zip(GRIDS, rows, strict=True))
        )
        shared = run(
            [ROOT / "build/multigrid/multigrid", *common, "--grids", spec, "--output", packed]
        )
        with packed.open("rb") as f:
            layout = read(f)
        if (
            shared["gpu"] != "NVIDIA GB10"
            or shared["projection_passes"] != 1
            or len(layout["regions"]) != len(GRIDS)
        ):
            raise ValueError("unexpected multi-grid execution")
        for (n, fov, box), row, region in zip(GRIDS, rows, layout["regions"], strict=True):
            target = tmp / f"{n}.f32"
            run(
                [
                    base / "cone-fdk",
                    *common,
                    "--n",
                    n,
                    "--fov",
                    fov,
                    "--roi",
                    row,
                    "--output",
                    target,
                ]
            )
            if (
                region["n"],
                region["fov_mm"],
                region["origin_xyz"] + region["shape_xyz"],
                region["sha256"],
            ) != (n, fov, list(box), digest(target)) or target.stat().st_size != region[
                "size_bytes"
            ]:
                raise ValueError(f"grid {n}: differs from standalone reconstruction")
        return {
            "device": shared["gpu"],
            "views": 90,
            "projection_passes": 1,
            "matched_standalone": len(GRIDS),
            "pack_sha256": layout["pack_sha256"],
        }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2))
