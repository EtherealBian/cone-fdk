"""Calibration-only finite-state segmentation of the familiar development interval."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from pore_axial_raw import (
    DEV,
    PTX,
    ROOT,
    SETTINGS,
    SHAPE,
    Denoise,
    Z,
    digest,
    sha,
    stack,
    token,
    verified,
    write,
)
from pore_byte_window_gpu import classify
from pore_patch_stats import median_stack
from pore_region import EV
from pore_transfer import SPEC
from scipy import ndimage as ndi

ARMS = ("source", "relative", "exact")
E = EV / "byte-window"


def identity():
    files = (
        "pores/BYTE-WINDOW-TABLE.md",
        "pores/byte_window.cu",
        "scripts/pore_byte_window.py",
        "scripts/pore_byte_window_gpu.py",
        "scripts/pore_byte_window_score.py",
        "pores/local_model.py",
        "pores/denoise.py",
        "scripts/pore_axial_raw.py",
        "scripts/pore_axial_score.py",
        "scripts/pore_transfer.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_section_score.py",
        "pores/pore_check.py",
        "pores/validate.py",
    )
    return dict(
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in files
        },
        compiled={
            k: sha(PTX / k)
            for k in ("byte-window.ptx", "search-support.ptx", "pores.ptx", "cone-pores.exe")
        },
    )


def encode(ct, lo, u, hi):
    assert (
        0 <= ct < 128
        and lo.dtype == u.dtype == hi.dtype == np.uint8
        and lo.shape == u.shape == hi.shape
        and np.all(lo <= u)
        and np.all(u <= hi)
    )
    return np.ascontiguousarray(
        ((ct << 24) | (lo.astype("i4") << 16) | (u.astype("i4") << 8) | hi)
    )


def decode(k):
    ct = k >> 24
    low = (k >> 16) & 255
    u = (k >> 8) & 255
    high = k & 255
    mid = (high + low) // 2
    span = high - low
    r = ((mid < 128).astype("i4") * 512 + span - ct + 255) * 512 + (mid - u) + 255
    return r.astype("i4"), np.where(span < ct, mid < 128, u < mid)


def graykeys(raw, i, z, gpu):
    if i not in DEV or not (160 <= z and z + 4 <= 184 or Z[0] <= z and z + 4 <= Z[1]):
        raise ValueError("Not registered")
    scale, shift, ct = SETTINGS[i]
    med = median_stack(raw, i, z, z + 4, 26)
    v = (gpu(med, 30 / scale) + shift) * scale
    u = np.clip(np.floor(v + 0.5), 0, 255).astype("u1")
    axis = np.arange(-5, 6)
    footprint = (axis[:, None] ** 2 + axis[None, :] ** 2 <= 26)[None]
    crop = np.s_[:, 26:-26, 26:-26]
    a = u[crop]
    lo = ndi.minimum_filter(u, footprint=footprint)[crop]
    hi = ndi.maximum_filter(u, footprint=footprint)[crop]
    assert a.shape == (4, 256, 256)
    mn = np.full_like(a, 255)
    mx = np.zeros_like(a)
    for dy, dx in zip(*np.where(footprint[0]), strict=True):
        v = u[:, 21 + dy : 277 + dy, 21 + dx : 277 + dx]
        np.minimum(mn, v, out=mn)
        np.maximum(mx, v, out=mx)
    assert int(footprint.sum()) == 89 and np.array_equal(mn, lo) and np.array_equal(mx, hi)
    key = encode(ct, lo, a, hi)
    _, m = decode(key)
    mid = (mn.astype("i4") + mx) // 2
    assert np.array_equal(m, np.where(mx.astype("i4") - mn < ct, mid < 128, mid > a))
    return key


def census(keys, t):
    assert keys.shape == t.shape and t.dtype == np.bool_
    k, inv, n = np.unique(keys.ravel(), return_inverse=True, return_counts=True)
    hit = np.bincount(inv[t.ravel()], minlength=len(k))
    return k.astype("i4"), n.astype("i4"), hit.astype("i4")


def lookup(q, base, k, v):
    m = np.searchsorted(k, q)
    j = np.minimum(m, len(k) - 1)
    hit = (m < len(k)) & (q == k[j])
    d = np.where(hit, v[j], 0) + np.where(base, 8, -8)
    return np.where(d == 0, base, d > 0)


def predict(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    ident = identity()
    rec = {}
    kk = []
    yy = []
    sources = {}
    with Denoise(PTX / "search-support.ptx") as gpu:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            sources[i] = raw
            k = np.concatenate([graykeys(raw, i, z, gpu) for z in range(160, 184, 4)])
            y = stack(seg, i, "segmented", 160, 184)
            kk.append(k.ravel())
            yy.append(y.ravel())
            rec[str(i)] = dict(
                raw=rs["sha256"],
                segmented=ss["sha256"],
                spacing_xyz_um=[SPEC[i][2]] * 3,
                calibration_tuples=digest(k),
                calibration_labels=digest(y),
                calibration_count=y.size,
            )
        ks = np.concatenate(kk)
        t = np.concatenate(yy)
        del kk, yy
        rs, _ = decode(ks)
        rr = census(rs, t)
        ee = census(ks, t)
        params = (rr[0], 2 * rr[2] - rr[1], ee[0], 2 * ee[2] - ee[1])
        model = out / "counts.npz"
        np.savez_compressed(
            model,
            **{
                prefix + name: a
                for prefix, s in [("r", rr), ("e", ee)]
                for name, a in zip(("key", "n", "positive"), s, strict=True)
            },
        )
        del rs, t, ks
        for i in DEV:
            hh = hashlib.sha256()
            masks = {k: np.empty(SHAPE, bool) for k in ARMS}
            exseen = 0
            for z in range(*Z, 4):
                x = graykeys(sources[i], i, z, gpu)
                hh.update(x.view("u1"))
                q = x.ravel()
                rel, src = decode(q)
                a = lookup(rel, src, *params[:2])
                b = lookup(q, a, *params[2:])
                native = classify(q, params)
                exseen += int(np.isin(q, ee[0], kind="sort").sum())
                enc = src.astype("u1") + 2 * a.astype("u1") + 4 * b.astype("u1")
                assert np.array_equal(enc, native)
                for bit, k in enumerate(ARMS):
                    masks[k][z - Z[0] : z - Z[0] + 4] = ((native >> bit) & 1).reshape(x.shape)
            r = rec[str(i)]
            r.update(transfer_tuples=hh.hexdigest(), seen_exact=exseen, masks={})
            for k, m in masks.items():
                p = out / f"sample{i}-{k}.npz"
                np.savez_compressed(p, mask=m)
                r["masks"][k] = dict(file=sha(p), mask=digest(m), cpu_gpu_differences=0)
            print(i, exseen, flush=True)
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            counts=sha(model),
            calibration_keys={"relative": len(rr[0]), "exact": len(ee[0])},
            shape_zyx=list(SHAPE),
            origin_xyz=[360, 376, Z[0]],
            specimens=rec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    predict(a.cache, a.out)
