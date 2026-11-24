"""Post-scoring topology diagnostics; predictions never consume this module or reference graphs."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pores"))
from audit import decode  # noqa: E402
from cases import eligible  # noqa: E402


def analyze(pred, truth):
    """Enumerate exact overlap edges, including contacts to clipped/tiny neighbors."""
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError("Expected same-shape 3-D masks")
    # Dense relabeling avoids allocating a histogram indexed by stable raw voxel ID.
    lp, npred = ndi.label(pred)
    lr, nref = ndi.label(truth)
    active = lp > 0
    first = np.full(npred + 1, lp.size, dtype="i8")
    np.minimum.at(first, lp[active], np.flatnonzero(active))
    first += 1
    pcounts = np.bincount(lp.ravel(), minlength=npred + 1)
    rcounts = np.bincount(lr.ravel(), minlength=nref + 1)
    ep, er = eligible(lp), eligible(lr)
    keys, kount = np.unique(
        lp[active & truth].astype("i8") * (nref + 1) + lr[active & truth], return_counts=True
    )
    edges = []
    p_adj = {}
    r_adj = {}
    for k, c in zip(keys, kount, strict=True):
        p, r = divmod(int(k), nref + 1)
        c = int(c)
        p_adj.setdefault(p, []).append((r, c))
        r_adj.setdefault(r, []).append((p, c))
    for p, adj in p_adj.items():
        for r, c in adj:
            if r not in er and p not in ep:
                continue
            edges.append(
                (
                    int(first[p]),
                    r,
                    int(pcounts[p]),
                    int(rcounts[r]),
                    c,
                    p in ep,
                    r in er,
                    sum(4 * m >= rcounts[q] and rcounts[q] >= 64 for q, m in adj),
                    sum(4 * m >= pcounts[q] and pcounts[q] >= 64 for q, m in r_adj[r]),
                )
            )
    # Include eligible components with no overlapping foreground at all.
    for p in sorted(ep - set(p_adj)):
        edges.append((int(first[p]), 0, int(pcounts[p]), 0, 0, True, False, 0, 0))
    for r in sorted(er - set(r_adj)):
        edges.append((0, r, 0, int(rcounts[r]), 0, False, True, 0, 0))
    return sorted(edges)


def export(directory, i):
    base = directory / f"sample{i}-review.html"
    text = base.read_text(encoding="utf8")
    data = json.loads(
        re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S)[1]
    )
    del text
    nz, ny, nx = data["shape"]
    if len(data["layers"]) != nz:
        raise ValueError("Layer count differs from the declared shape")
    masks = []
    for k, is_binary in [("objects", False), ("reference", True)]:
        masks.append(
            np.stack(
                [
                    (decode(layer[k], ny * nx, is_binary) != 0).reshape(ny, nx)
                    for layer in data["layers"]
                ]
            )
        )
    rows = analyze(*masks)
    with (directory / f"sample{i}-overlap.csv").open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pred_id",
                "ref_label",
                "pred_voxels",
                "ref_voxels",
                "intersection_voxels",
                "pred_eligible",
                "ref_eligible",
                "n_ref_covered_quarter_by_pred",
                "n_pred_covered_quarter_by_ref",
            ]
        )
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("specimens", nargs="+", type=int, choices=[2, 3, 5])
    a = ap.parse_args()
    for i in a.specimens:
        print(i, len(export(a.directory, i)), "overlap rows")
