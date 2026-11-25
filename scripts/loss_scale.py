"""Known-topology wide cylinder, generated one wall row at a time, not a projection accuracy test."""

from pathlib import Path

import numpy as np
from compare_native import sha
from loss import inspect_loss
from loss_oracle import unpack
from validate_cylinder import spec
from validate_roi import require
from validate_stack import save_json
from wall import WALL_DTYPE


def qualify_scale(base: Path, dest: Path):
    import json

    w, h = 4096, 1024
    g = spec(base / "wide.grid", (w, h, 101), (0, 0, 0), (0, 0, 1), (1, 0, 0), (4, 0.06, 0.1))
    reference = base / "wide-reference.wall"
    part = base / "wide-inspected.wall"
    out = base / "wide.json"
    row = np.zeros(w, dtype=WALL_DTYPE)
    row[:] = (5, 9, 4, 0, 0)
    # Two known periodic components; one contains a one-cell unknown, the other touches both axial edges.
    with reference.open("wb") as a, part.open("wb") as b:
        for y in range(h):
            row.tofile(a)
            current = row.copy()
            current[120:129] = (5.5, 9, 3.5, 0, 0)
            if 20 <= y < h - 20:
                for sl in (slice(0, 12), slice(-12, None)):
                    current[sl] = (5, 8, 3, 0, 0)
            if y == h // 2:
                current[0] = (np.nan, np.nan, np.nan, 3, 0)
            current.tofile(b)
    small = inspect_loss(reference, g, part, g, out, 0.30, 9, budget=1)
    digest = sha(out)
    large = inspect_loss(reference, g, part, g, out, 0.30, 9, budget=16)
    require(
        digest == sha(out) and small["tiles"] > large["tiles"] > 1,
        "wide map replay did not span tiles",
    )
    require(
        all(
            r["redzones_verified"]
            and r["input_bytes_read"] == 4 * reference.stat().st_size
            and r["peak_owned_bytes"] <= r["budget_bytes"]
            for r in (small, large)
        ),
        "wide map byte accounting",
    )
    doc = json.loads(out.read_bytes())
    labels = unpack(doc)
    target = np.zeros((h, w), np.int32)
    target[:, 120:129] = 121
    for sl in (slice(0, 12), slice(-12, None)):
        target[20:-20, sl] = 20 * w + 1
    target[h // 2, 0] = -5
    np.testing.assert_array_equal(labels, target)
    first, second = doc["regions"]
    require(
        first["touches_axial_edge"]
        and not first["touches_unknown"]
        and not first["crosses_seam"]
        and second["crosses_seam"]
        and second["touches_unknown"]
        and not second["touches_axial_edge"],
        "wide region flags",
    )
    require(
        doc["coverage"]["unknown"] == 1 and doc["coverage"]["flagged"] == int((target > 0).sum()),
        "wide coverage",
    )
    save_json(
        dest / "wall-loss-scale.json",
        dict(
            protocol="Generated wall records with a closed-form footprint; not an X-ray measurement or a speed benchmark.",
            shape=[h, w],
            wall_pair_bytes=2 * reference.stat().st_size,
            reference_sha256=sha(reference),
            inspected_sha256=sha(part),
            report_sha256=digest,
            all_labels_match=True,
            regions=doc["regions"],
            coverage=doc["coverage"],
            one_mib=small,
            sixteen_mib=large,
        ),
    )
