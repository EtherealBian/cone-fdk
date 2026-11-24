"""Read-only bridge review queue. --probability needs just a 3-D NPY of probabilities."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_transfer import DEV, label
from pore_transfer_replay import read_npz
from scipy import ndimage as ndi


def flag(prob):
    if (
        prob.ndim != 3
        or prob.size == 0
        or prob.dtype.kind != "f"
        or not np.isfinite(prob).all()
        or prob.min() < 0
        or prob.max() > 1
    ):
        raise ValueError("Expected a nonempty finite 3-D probability volume in [0,1]")
    lab, vol, whole, boxes = label(prob >= 0.5)
    rows = []
    for p in np.flatnonzero(whole & (vol >= 64)):
        box = boxes[p - 1]
        part = lab[box] == p
        seed = tuple(int(v[0] + t.start) for v, t in zip(np.nonzero(part), box, strict=True))
        sub, n = ndi.label(part & (prob[box] >= 0.90))
        sizes = np.bincount(sub.ravel(), minlength=n + 1)[1:]
        cores = sorted(
            (int(k) for k in sizes if k >= 32 and int(k) * 20 >= int(vol[p])), reverse=True
        )
        rows.append(
            dict(
                pred=int(p),
                native_id=int(np.ravel_multi_index(seed, prob.shape)) + 1,
                seed_zyx=list(seed),
                voxels=int(vol[p]),
                core_sizes=cores,
                flagged=len(cores) >= 2,
            )
        )
    return lab, rows


def compare(lab, rows, truth):
    ref, v, w, _ = label(truth)
    keep = w & (v >= 64)
    both = (lab > 0) & keep[ref]
    keys, hits = np.unique(lab[both].astype("i8") * len(v) + ref[both], return_counts=True)
    parents = {r["pred"]: [] for r in rows}
    for key, h in zip(keys, hits, strict=True):
        p, r = divmod(int(key), len(v))
        if p in parents and 2 * h >= v[r]:
            parents[p].append(dict(reference=r, reference_voxels=int(v[r]), hits=int(h)))
    tally = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    audit = []
    for row in rows:
        overlaps = parents[row["pred"]]
        multi = len(overlaps) >= 2
        k = ("tp" if multi else "fp") if row["flagged"] else ("fn" if multi else "tn")
        tally[k] += 1
        audit.append(dict(**row, covered_references=overlaps, event=multi, category=k))
    return dict(counts=tally, eligible=len(rows), rows=audit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probability", type=Path)
    g.add_argument("--packs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.probability is not None:
        _, rows = flag(np.load(a.probability, allow_pickle=False))
        result = dict(rows=rows)
    else:
        result = {}
        for i in DEV:
            p, _, _, t = read_npz(a.packs / f"sample{i}-review.npz")
            lp, rows = flag(p)
            result[str(i)] = compare(lp, rows, t)
            print(i, result[str(i)]["counts"])
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
