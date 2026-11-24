"""Trace a paired section-screen miss against the published masks and source reference."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pore_section_score import ARMS, DEV, ROOT, SEAL, SHAPE, Z, digest, stack, verified
from scipy import ndimage as ndi


def clipped(labels, k):
    return any(
        (np.take(labels, side, axis=axis) == k).any() for axis in range(3) for side in (0, -1)
    )


def trace(cache, archive, specimen, ref):
    if specimen not in DEV:
        raise ValueError("Development only")
    m = json.loads(SEAL.read_text())
    s = m["specimens"][str(specimen)]
    seg, ss = verified(cache, specimen, "segmented", False)
    if s["segmented"] != ss["sha256"]:
        raise ValueError("Unexpected reference")
    t, total = ndi.label(stack(seg, specimen, "segmented", *Z))
    at = np.flatnonzero(t.ravel() == ref)
    n = len(at)
    if not 0 < ref <= total or n < 64 or clipped(t, ref):
        raise ValueError("Expected eligible reference")
    del t
    r = dict(
        specimen=specimen,
        reference_id=ref,
        reference_voxels=n,
        segmented=ss["sha256"],
        arms={},
    )
    with ZipFile(archive) as z:
        for arm in ARMS:
            data = z.read(f"sample{specimen}-{arm}.npy")
            a = np.load(io.BytesIO(data), allow_pickle=False)
            locks = s["masks"][arm]
            if (
                a.dtype != bool
                or a.shape != SHAPE
                or hashlib.sha256(data).hexdigest() != locks["file"]
                or digest(a) != locks["mask"]
            ):
                raise ValueError("Changed mask")
            lab, _ = ndi.label(a)
            counts = np.bincount(lab.ravel()[at])
            rows = []
            for q in np.flatnonzero(counts):
                if not q:
                    continue
                c = int(counts[q])
                size = int(sum(np.count_nonzero(plane == q) for plane in lab))
                u = n + size - c
                rows.append(
                    dict(
                        scipy_label=int(q),
                        voxels=size,
                        unclipped=not clipped(lab, q),
                        intersection=c,
                        union=u,
                        iou=c / u,
                    )
                )
            r["arms"][arm] = sorted(rows, key=lambda v: (-v["iou"], v["scipy_label"]))
            del lab, a, data
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument(
        "--masks", type=Path, default=ROOT / "pores/evidence/section-area/weights-masks.zip"
    )
    ap.add_argument("--specimen", required=True, type=int, choices=DEV)
    ap.add_argument("--ref", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    r = trace(a.cache, a.masks, a.specimen, a.ref)
    text = json.dumps(r, indent=2, allow_nan=False)
    with a.out.open("x") as f:
        f.write(text)
        f.write("\n")
