"""Seal exact local byte flip radii; evaluate only optimistic voxel bounds, not new masks."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pore_bernsen_provenance import quantize
from pore_patch_stats import (
    DEV,
    ROOT,
    SETTINGS,
    Denoise,
    H,
    decision,
    median_stack,
    score,
    stack,
    verified,
)
from scipy import ndimage as ndi

SEAL = ROOT / "pores/evidence/byte-radius/predictions.json"
SHAPE = (128, 256, 256)
LEVELS = (0, 1, 2, 4, 8)


def array_sha(v):
    return hashlib.sha256(np.ascontiguousarray(v).view("u1")).hexdigest()


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def possibilities(c, neighbor_min, h, ct, e):
    # Independent enumeration of all three integer extrema, including center.
    def values(v):
        return np.arange(max(0, v - e), min(255, v + e) + 1, dtype="i2")

    C, L, H = (
        values(c)[:, None, None],
        values(neighbor_min)[None, :, None],
        values(h)[None, None, :],
    )
    valid = L <= H
    low = np.minimum(C, L)
    high = np.maximum(C, H)
    mid = (low + high) // 2
    p = np.where(high - low < ct, mid < 128, C < mid)
    return bool((valid & p).any()), bool((valid & ~p).any())


def exact(c, neighbor_min, h, ct):
    base = possibilities(c, neighbor_min, h, ct, 0)[0]
    for e in range(1, 9):
        p, s = possibilities(c, neighbor_min, h, ct, e)
        if s if base else p:
            return e if base else -e
    return 9 if base else -9


def save(path, obj):
    text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf8") as f:
        f.write(text)


def predict(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    rng = np.random.default_rng(61429)
    pts = [(dy, dx) for dy in range(-5, 6) for dx in range(-5, 6) if 0 < dy * dy + dx * dx <= 26]
    src = ROOT / "scripts/bernsen_radius.cu"
    ptx = ROOT / "build/pores/bernsen-radius.ptx"
    nlm = ROOT / "build/pores/search-support.ptx"
    with Denoise(nlm) as smoother, Denoise(ptx) as radius:
        for i in DEV:
            path, rs = verified(cache, i, "raw", False)
            ct = SETTINGS[i][2]
            maps = np.empty(SHAPE, dtype="i1")
            bytehash = hashlib.sha256()
            samples = []
            for a in range(400, 528, 2):
                m = median_stack(path, i, a, a + 2, H)
                den = smoother(m, 30 / SETTINGS[i][0])
                u = quantize(den, i)
                bytehash.update(u.tobytes())
                p = decision(den, i, 0)[:, H:-H, H:-H]
                v = radius(u, ct)[:, H:-H, H:-H]
                codes = v.astype("i1")
                if (
                    not np.array_equal(v, codes)
                    or not np.array_equal(codes > 0, p)
                    or not np.isin(np.abs(codes), np.arange(1, 10)).all()
                ):
                    raise ValueError("Invalid radius or changed baseline")
                maps[a - 400 : a - 398] = codes
                # Include six random points and representatives of both signs when present.
                take = rng.choice(codes.size, 6, replace=False).tolist()
                for match in (codes > 0, codes < 0, np.abs(codes) == 1, np.abs(codes) == 9):
                    at = np.flatnonzero(match)
                    if len(at):
                        take.append(int(at[rng.integers(len(at))]))
                for at in sorted(set(take)):
                    z, y, x = map(int, np.unravel_index(at, codes.shape))
                    neigh = [int(u[z, y + H + j, x + H + k]) for j, k in pts]
                    c = int(u[z, y + H, x + H])
                    lo, hi = min(neigh), max(neigh)
                    n = exact(c, lo, hi, ct)
                    if n != codes[z, y, x]:
                        raise ValueError(
                            f"CUDA radius differs from independent enumeration: {i, a, z, y, x, c, lo, hi, n, codes[z, y, x]}"
                        )
                    samples.append([a + z, y + 376, x + 360, c, lo, hi, n])
            path = out / f"sample{i}.npz"
            np.savez_compressed(path, signed=maps)
            records[i] = dict(
                raw=rs["sha256"],
                file=sha(path),
                signed=array_sha(maps),
                baseline=array_sha(maps > 0),
                halo_byte_stream=bytehash.hexdigest(),
                sampled_patch_checks=samples,
                counts={str(j): int((maps == j).sum()) for j in (*range(-9, 0), *range(1, 10))},
            )
            print(i, "sealed", len(samples), flush=True)
    return dict(
        origin_xyz=[360, 376, 400],
        shape_zyx=SHAPE,
        levels=LEVELS,
        kernel_sha256=hashlib.sha256(src.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        radius_ptx=sha(ptx),
        nlm_ptx=sha(nlm),
        specimens=records,
    )


def bounds(codes, t, e):
    # A perfect oracle chooses truth at every individually changeable location.
    # No single gray volume is asserted to realize those choices simultaneously.
    fixed = np.abs(codes) > e
    fp = (codes > 0) & ~t & fixed
    fn = (codes < 0) & t & fixed
    z = score((t & ~fn) | fp, t)
    return dict(
        unavoidable_fp=int(np.count_nonzero(fp)),
        unavoidable_fn=int(np.count_nonzero(fn)),
        dice_upper=z["dice"],
        precision_upper=z["precision"],
        recall_upper=z["recall"],
    )


def evaluate(cache, pred, out):
    d = json.loads(SEAL.read_text())
    out.mkdir(parents=True, exist_ok=False)
    if set(d["specimens"]) != set("235") or d != json.loads(
        (pred / "predictions.json").read_text()
    ):
        raise ValueError("Not the public seal")
    for i in DEV:
        p = pred / f"sample{i}.npz"
        if sha(p) != d["specimens"][str(i)]["file"]:
            raise ValueError("Unsealed input")
    results = {}
    for i in DEV:
        with np.load(pred / f"sample{i}.npz") as f:
            codes = f["signed"]
        r = d["specimens"][str(i)]
        if (
            codes.shape != SHAPE
            or codes.dtype != "i1"
            or array_sha(codes) != r["signed"]
            or array_sha(codes > 0) != r["baseline"]
        ):
            raise ValueError("Different signed map")
        seg, ss = verified(cache, i, "segmented", False)
        t = stack(seg, i, "segmented", 400, 528)
        lab, n = ndi.label(t)
        sizes = np.zeros(n + 1, dtype="i8")
        for plane in lab:
            sizes += np.bincount(plane.ravel(), minlength=n + 1)
        clipped = set(
            np.unique(
                np.concatenate(
                    [np.take(lab, side, axis=ax).ravel() for ax in range(3) for side in (0, -1)]
                )
            )
        )
        refs = sorted(set(np.flatnonzero(sizes >= 64)) - clipped - {0})
        rows = {int(k): {"reference_id": int(k), "voxels": int(sizes[k])} for k in refs}
        levels = {}
        for e in LEVELS:
            levels[e] = bounds(codes, t, e)
            counts = np.bincount(lab[(codes < -e) & t], minlength=n + 1)
            for k in refs:
                rows[k][f"locked_fn_at_{e}"] = int(counts[k])
        cp = out / f"sample{i}-references.csv"
        with cp.open("x", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["reference_id", "voxels", *[f"locked_fn_at_{e}" for e in LEVELS]]
            )
            w.writeheader()
            w.writerows(rows.values())
        results[i] = dict(
            segmented=ss["sha256"],
            baseline=score(codes > 0, t),
            levels=levels,
            references=len(rows),
            references_csv=sha(cp),
        )
        print(i, levels, flush=True)
        del t, lab, codes
    save(out / "results.json", results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["predict", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--pred", type=Path)
    a = ap.parse_args()
    if a.phase == "predict":
        save(a.out / "predictions.json", predict(a.cache, a.out))
    elif a.pred:
        evaluate(a.cache, a.pred, a.out)
    else:
        ap.error("Score needs --pred")
