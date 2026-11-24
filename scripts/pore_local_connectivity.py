"""Measure signed local six-neighbor connectivity on calibration labels only."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, ROOT, digest, stack, verified, write
from scipy import ndimage as ndi

POSITIONS = list(np.ndindex(3, 3, 3))
FACES = [j for j, p in enumerate(POSITIONS) if sum(abs(x - 1) for x in p) == 1]
FACE_MASK = np.uint32(sum(1 << j for j in FACES))


def critical(t):
    if t.ndim != 3 or min(t.shape) < 3 or t.dtype != bool:
        raise ValueError("Expected 3D binary pore mask")
    shape = tuple(n - 2 for n in t.shape)
    w = np.zeros(shape, "u4")
    for j, p in enumerate(POSITIONS):
        if j != 13:
            region = tuple(slice(i, i + n) for i, n in zip(p, shape, strict=True))
            w |= t[region].astype("u4") << j
    f = w & FACE_MASK
    possible = (f & (f - np.uint32(1))) != 0
    block, face = w[possible], f[possible]
    reached = face & (np.uint32(0) - face)
    steps = []
    for axis, shift in enumerate((9, 3, 1)):
        steps.append(
            (
                shift,
                *(
                    np.uint32(sum(1 << j for j, p in enumerate(POSITIONS) if p[axis] != edge))
                    for edge in (2, 0)
                ),
            )
        )
    for _ in range(26):
        adj = reached.copy()
        for stride, pos, neg in steps:
            adj |= ((reached & pos) << stride) | ((reached & neg) >> stride)
        adj &= block
        if np.array_equal(adj, reached):
            break
        reached = adj
    out = np.zeros_like(t)
    out[1:-1, 1:-1, 1:-1][possible] = (face & ~reached) != 0
    return out


def scalar(a):
    shell = a.copy()
    shell[1, 1, 1] = False
    labels, _ = ndi.label(shell)
    return len(set(labels.ravel()[FACES]) - {0}) >= 2


def describe(t, m):
    interior = np.zeros_like(t)
    interior[1:-1, 1:-1, 1:-1] = True
    rng = np.random.default_rng(61429)
    counts = {}
    audited = 0
    for flag in (False, True):
        for hit in (False, True):
            pool = np.argwhere(interior & (t == flag) & (m == hit))
            for z, y, x in pool[rng.choice(len(pool), size=min(128, len(pool)), replace=False)]:
                if scalar(t[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2]) != hit:
                    raise ValueError("Scalar shell replay disagrees")
                audited += 1
        per = (m & (t == flag)).sum((1, 2))
        counts["connector" if flag else "gap"] = dict(
            count=int(per.sum()), planes=int((per > 0).sum()), per_z=per.tolist()
        )
    return dict(
        signs=counts,
        eligible=all(r["count"] >= 32 and r["planes"] >= 4 for r in counts.values()),
        scalar_checks=audited,
        critical_sha256=digest(m),
    )


def census(cache):
    prior = json.loads((ROOT / "pores/evidence/census-train/fit.json").read_text())
    rows = {}
    for i in DEV:
        seg, ss = verified(cache, i, "segmented", False)
        t = stack(seg, i, "segmented", 160, 184)
        ref = prior["specimens"][str(i)]
        if ss["sha256"] != ref["segmented"] or digest(t) != ref["cal_mask"]:
            raise ValueError("Wrong calibration source")
        rows[i] = dict(segmented=ss["sha256"], cal_mask=digest(t), **describe(t, critical(t)))
        print(i, rows[i], flush=True)
    return dict(
        calibration_z=[160, 184],
        specimens=rows,
        eligible=all(r["eligible"] for r in rows.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for s in ("cache", "out"):
        p.add_argument("--" + s, type=Path, required=True)
    a = p.parse_args()
    write(a.out, census(a.cache))
