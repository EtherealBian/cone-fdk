"""Development-only bounded support growth; predictions never open a segmented archive."""

import argparse
import hashlib
import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, PTX, ROOT, SHAPE, Z, digest, sha, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_contrast_audit import shell
from pore_seed_gpu import median7
from pore_transfer import FACE, label
from scipy import ndimage as ndi

E = ROOT / "pores/evidence/small-seed"
ARMS = ("gray", "geometry")
NBR = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))


def identity():
    paths = (
        "scripts/pore_small_seed.py",
        "scripts/pore_seed_gpu.py",
        "scripts/pore_seed_median.cu",
        "scripts/pore_seed_score.py",
        "scripts/pore_contrast_audit.py",
        "scripts/pore_axial_score.py",
        "scripts/pore_transfer.py",
        "pores/local_model.py",
        "pores/validate.py",
        "pores/SMALL-SEED-PILOT.md",
    )
    return dict(
        files={
            f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for f in paths
        },
        ptx=sha(PTX / "seed-median.ptx"),
        census=sha(EVID / "predictions.json"),
    )


def fill(seed, allowed, cost, n=None):
    """Reachable support; geometry ties use x-fastest order within the fixed box."""
    if n is not None and n < 0:
        raise ValueError("Negative request")
    if n == 0:
        return np.zeros_like(seed)
    out = np.zeros_like(seed)
    seen = seed.copy()
    queue = deque() if cost is None else []

    def visit(q):
        seen[q] = True
        if cost is None:
            queue.append(q)
        else:
            heapq.heappush(queue, (cost[q], q))

    for v in np.argwhere(ndi.binary_dilation(seed, structure=FACE) & allowed):
        visit(tuple(v))
    count = 0
    while queue:
        q = queue.popleft() if cost is None else heapq.heappop(queue)[1]
        out[q] = True
        count += 1
        if n == count:
            break
        for step in NBR:
            u = tuple(x + y for x, y in zip(q, step, strict=True))
            if (
                all(0 <= v < w for v, w in zip(u, seed.shape, strict=True))
                and not seen[u]
                and allowed[u]
            ):
                visit(u)
    if n is not None and n != count:
        raise ValueError("Not enough reachable sites")
    return out


def recover(base, raw, med, cpu):
    lab, sizes, whole, boxes = label(base)
    if (
        raw.shape != tuple(n + 8 for n in lab.shape)
        or med.shape != base.shape
        or not np.array_equal(cpu, med)
    ):
        raise ValueError("Rank replay mismatch or wrong halo")
    shape = base.shape
    interior = np.zeros(shape, bool)
    interior[1:-1, 1:-1, 1:-1] = True
    masks = {k: base.copy() for k in ARMS}
    rows = []
    anchors = []
    for j, box in enumerate(boxes, 1):
        src = lab[box] == j
        u = np.unravel_index(src.argmax(), src.shape)
        anchor = tuple(int(s.start + x) for s, x in zip(box, u, strict=True))
        anchors.append(anchor)
        if not whole[j] or not 8 <= sizes[j] < 64:
            continue
        # Include three extra sites beyond the two-site corridor, for foreign supports.
        bb = tuple(
            slice(max(0, s.start - 5), min(n, s.stop + 5))
            for s, n in zip(box, shape, strict=True)
        )
        local = lab[bb]
        seed = local == j
        other = (local != 0) & ~seed
        d2 = np.rint(ndi.distance_transform_edt(~seed) ** 2).astype("i4")
        allowed = (d2 >= 1) & (d2 <= 4) & interior[bb] & ~other
        if other.any():
            allowed &= ndi.distance_transform_edt(~other) > 3
        pix = raw[tuple(slice(s.start, s.stop + 8) for s in box)]
        a = int(2 * np.median(pix[4:-4, 4:-4, 4:-4][src]))
        b = int(2 * np.median(pix[shell(src)]))
        bright = 4 * med[bb].astype("i4") <= a + b
        grow = (
            (ndi.binary_propagation(seed, structure=FACE, mask=seed | (allowed & bright)) & ~seed)
            if a < b
            else np.zeros_like(seed)
        )
        independent = fill(seed, allowed & (4 * cpu[bb].astype("i4") <= a + b) & (a < b), None)
        assert np.array_equal(grow, independent)
        count = int(grow.sum())
        equal = fill(seed, allowed, d2, count)
        for k, new in zip(ARMS, (grow, equal), strict=True):
            assert not (masks[k][bb] & new).any()
            masks[k][bb] |= new
        rows.append(
            dict(
                label=j,
                seed_id=int(np.ravel_multi_index(anchor, shape)) + 1,
                seed_zyx=anchor,
                original=int(sizes[j]),
                inside_median_x2=a,
                shell_median_x2=b,
                allowed=int(allowed.sum()),
                added=count,
                final=int(sizes[j]) + count,
                different_sites=int(np.count_nonzero(grow != equal)),
            )
        )
    increments = {r["label"]: r["added"] for r in rows}
    for k, m in masks.items():
        new, ns, nw, nb = label(m)
        idx = new[tuple(np.array(anchors, dtype=int).T)] if anchors else np.array([], int)
        assert (
            not (base & ~m).any()
            and len(set(idx)) == len(nb) == len(boxes)
            and np.array_equal(nw[idx], whole[1:])
        )
        assert np.array_equal(
            ns[idx],
            sizes[1:] + np.array([increments.get(j, 0) for j in range(1, len(boxes) + 1)]),
        )
    return masks, rows


def predict(cache, out):
    from local_model import context_stack

    out.mkdir(parents=True, exist_ok=False)
    parent = json.loads((EVID / "predictions.json").read_bytes())
    specs = {}
    for i in DEV:
        r = parent["specimens"][str(i)]
        base = read_mask(EVID, i, "census", r["masks"]["census"])
        raw, rs = verified(cache, i, "raw", False)
        assert rs["sha256"] == r["raw"]
        pix = context_stack(raw, i, Z[0] - 3, Z[1] + 3, 4)
        assert pix.shape == tuple(n + 8 for n in SHAPE)
        med = median7(pix)
        rank = ndi.rank_filter(pix, rank=3, footprint=FACE)[4:-4, 4:-4, 4:-4]
        masks, rows = recover(base, pix, med, rank)
        rowfile = out / f"sample{i}-seeds.json"
        write(rowfile, rows)
        rec = {}
        for k, m in masks.items():
            f = out / f"sample{i}-{k}.npz"
            with f.open("xb") as fp:
                np.savez_compressed(fp, mask=m)
            rec[k] = dict(file=sha(f), mask=digest(m))
        specs[str(i)] = dict(
            raw=rs["sha256"],
            segmented=r["segmented"],
            census=digest(base),
            raw_halo=digest(pix),
            median7_u16=digest(med),
            median_cpu_differences=int(np.count_nonzero(med != rank)),
            growth_replay_differences=0,
            seeds_sha256=sha(rowfile),
            masks=rec,
        )
        print(
            i,
            "seeds",
            len(rows),
            "entrants",
            sum(q["final"] >= 64 for q in rows),
            "added",
            sum(q["added"] for q in rows),
            flush=True,
        )
    ident = identity()
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            specimens=specs,
            origin_xyz=[360, 376, Z[0]],
            shape_zyx=list(SHAPE),
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Do not run with -O")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    predict(a.cache, a.out)
