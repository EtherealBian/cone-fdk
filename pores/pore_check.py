"""Independent CPU reference: compare all components, not just total foreground."""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/pores/cone-pores.exe"


def command(src, out, shape, spacing, threshold, conn, slab, filt, budget):
    z, y, x = shape
    return list(
        map(
            str,
            [
                EXE,
                "--input",
                src,
                "--output",
                out,
                "--nx",
                x,
                "--ny",
                y,
                "--nz",
                z,
                "--sx",
                spacing[0],
                "--sy",
                spacing[1],
                "--sz",
                spacing[2],
                "--threshold",
                threshold,
                "--connectivity",
                conn,
                "--slab",
                slab,
                "--filter",
                filt,
                "--budget-mib",
                budget,
            ],
        )
    )


def native(src, out, shape, spacing, t, conn=6, slab=1000000, filt="none", budget=64):
    p = subprocess.run(
        command(src, out, shape, spacing, t, conn, slab, filt, budget), capture_output=True
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode("utf8", errors="replace"))
    return json.loads(p.stdout)


def reference(mask, conn, spacing, csv_file, stats):
    lab, n = ndi.label(mask, ndi.generate_binary_structure(3, 1 if conn == 6 else 3))
    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    np.testing.assert_equal(len(rows), n)
    z, y, x = np.nonzero(mask)
    ids = lab[z, y, x]
    counts = np.bincount(ids, minlength=n + 1)
    faces = np.zeros(n + 1, np.uint8)
    for axis, a in enumerate((x, y, z)):
        faces[ids[a == 0]] |= 1 << (2 * axis)
        faces[ids[a == mask.shape[2 - axis] - 1]] |= 2 << (2 * axis)
    assert stats["roi_voxels"] == mask.size and stats["pore_voxels"] == len(z)
    assert stats["components"] == n and stats["interior_components"] == int(
        (faces[1:] == 0).sum()
    )
    assert stats["interior_pore_voxels"] == int(counts[faces == 0].sum())
    assert stats["spanning_xyz"] == [
        int((((faces[1:] >> (2 * k)) & 3) == 3).sum()) for k in range(3)
    ]
    np.testing.assert_allclose(stats["porosity"], len(z) / mask.size, rtol=1e-13)
    if not n:
        return lab, rows

    def col(key, dtype):
        return np.array([r[key] for r in rows], dtype=dtype)

    first = np.full(n + 1, mask.size, np.int64)
    np.minimum.at(first, ids, np.ravel_multi_index((z, y, x), mask.shape))
    ordered = np.argsort(first[1:]) + 1
    np.testing.assert_array_equal(col("id", "i8"), first[ordered] + 1)
    for key, a in zip(
        ("seed_z", "seed_y", "seed_x"), np.unravel_index(first[ordered], mask.shape), strict=True
    ):
        np.testing.assert_array_equal(col(key, "i8"), a)
    np.testing.assert_array_equal(col("voxels", "i8"), counts[ordered])
    np.testing.assert_array_equal(col("faces", "i8"), faces[ordered])
    volume = counts[ordered] * np.prod(spacing)
    np.testing.assert_allclose(col("volume_um3", "f8"), volume, rtol=1e-12)
    np.testing.assert_allclose(
        col("equivalent_diameter_um", "f8"), np.cbrt(6 * volume / np.pi), rtol=1e-12
    )
    for axis, a, d in zip("xyz", (x, y, z), spacing, strict=True):
        centers = (
            np.bincount(ids, weights=a, minlength=n + 1)[ordered] / counts[ordered] + 0.5
        ) * d
        np.testing.assert_allclose(col("centroid_" + axis + "_um", "f8"), centers, rtol=1e-12)
    return lab, rows
