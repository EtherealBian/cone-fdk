"""Size-stratified diagnostics, never a replacement for the registered, unstratified gates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def stratify(folder, i):
    stem = f"sample{i}"
    report = json.loads((folder / (stem + ".json")).read_text(encoding="utf8"))
    spacing = report["native"]["spacing_xyz_um"]
    voxel_volume = float(np.prod(spacing))
    with (folder / (stem + "-matches.csv")).open(newline="") as f:
        matches = list(csv.DictReader(f))
    with (folder / (stem + "-size-bands.csv")).open("w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "reference_voxel_bin",
                "matches",
                "max_abs_volume_error_pct",
                "p95_abs_volume_error_pct",
                "median_abs_diameter_diff_um",
                "p95_abs_diameter_diff_um",
                "p95_centroid_diff_um",
            ]
        )
        for lo, hi in [(64, 128), (128, 256), (256, 512), (512, None), (64, None)]:
            ss = [
                r
                for r in matches
                if int(r["ref_voxels"]) >= lo and (hi is None or int(r["ref_voxels"]) < hi)
            ]
            if ss:
                err = np.array([abs(float(r["signed_volume_error_pct"])) for r in ss])
                diam = [
                    abs(
                        np.cbrt(6 * voxel_volume * int(r["pred_voxels"]) / np.pi)
                        - np.cbrt(6 * voxel_volume * int(r["ref_voxels"]) / np.pi)
                    )
                    for r in ss
                ]
                if not np.allclose(spacing, spacing[0], rtol=0, atol=0):
                    raise ValueError(
                        "Centroid µm conversion needs the original displacement vector for anisotropic voxels"
                    )
                center = [float(r["centroid_displacement_vox"]) * spacing[0] for r in ss]
                v = [
                    np.max(err),
                    np.percentile(err, 95),
                    np.median(diam),
                    np.percentile(diam, 95),
                    np.percentile(center, 95),
                ]
            else:
                v = [""] * 5
            writer.writerow([f"[{lo},{hi if hi is not None else 'inf'})", len(ss), *v])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("specimens", type=int, nargs="+")
    args = ap.parse_args()
    for i in args.specimens:
        stratify(args.folder, i)
