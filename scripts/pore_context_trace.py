"""Reconstruct every intersecting component for a reference on a sealed research slab."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, ROOT, SHAPE, Z, digest, sha, stack, verified, write
from scipy import ndimage as ndi


def overlaps(labels, region, mask, n):
    sizes = np.bincount(labels.ravel())
    boxes = ndi.find_objects(labels)
    counts = np.bincount(labels[region][mask])
    rows = []
    for j in np.flatnonzero(counts[1:]) + 1:
        hit = int(counts[j])
        size = int(sizes[j])
        box = boxes[j - 1]
        union = n + size - hit
        rows.append(
            dict(
                scipy_label=int(j),
                voxels=size,
                intersection=hit,
                union=union,
                iou=hit / union,
                unclipped=all(
                    s.start > 0 and s.stop < lim for s, lim in zip(box, labels.shape, strict=True)
                ),
                bbox_zyx=[[s.start, s.stop] for s in box],
            )
        )
    return sorted(rows, key=lambda v: (-v["iou"], v["scipy_label"]))


def trace(cache, study, i, ref):
    if study not in ("axial-raw", "census-train", "local-connectivity") or i not in DEV or ref <= 0:
        raise ValueError("Development trace only")
    d = ROOT / "pores/evidence" / study
    locked = json.loads((d / "predictions.json").read_text())
    rec = locked["specimens"][str(i)]
    seg, ss = verified(cache, i, "segmented", False)
    if (
        rec["segmented"] != ss["sha256"]
        or list(SHAPE) != locked["shape_zyx"]
        or locked["origin_xyz"] != [360, 376, Z[0]]
    ):
        raise ValueError("Changed reference or crop")
    truth, num = ndi.label(stack(seg, i, "segmented", *Z))
    if ref > num:
        raise ValueError("Unknown reference ID")
    box = ndi.find_objects(truth)[ref - 1]
    mask = truth[box] == ref
    n = int(mask.sum())
    if n < 64 or any(s.start == 0 or s.stop == lim for s, lim in zip(box, SHAPE, strict=True)):
        raise ValueError("Not an eligible reference")
    del truth
    arms = {}
    for k, v in rec["masks"].items():
        p = d / f"sample{i}-{k}.npz"
        if sha(p) != v["file"]:
            raise ValueError("Changed file")
        with np.load(p, allow_pickle=False) as f:
            m = f["mask"]
        if m.dtype != bool or m.shape != SHAPE or digest(m) != v["mask"]:
            raise ValueError("Changed mask")
        lab, _ = ndi.label(m)
        arms[k] = overlaps(lab, box, mask, n)
        del lab
    return dict(
        specimen=i,
        study=study,
        reference_id=ref,
        reference_voxels=n,
        reference_bbox_local_zyx=[[s.start, s.stop] for s in box],
        origin_xyz=locked["origin_xyz"],
        segmented=ss["sha256"],
        arms=arms,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--study", required=True, choices=("axial-raw", "census-train", "local-connectivity"))
    ap.add_argument("--specimen", type=int, required=True, choices=DEV)
    ap.add_argument("--ref", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    write(a.out, trace(a.cache, a.study, a.specimen, a.ref))
