"""Measure oriented pore faces from true raw line and plane profiles, with integer CUDA replay."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, PTX, ROOT, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_face_gpu import native
from pore_region import EV, read
from pore_transfer import FACE, label
from scipy import ndimage as ndi

BASE = EV / "volume-curvature"
E = EV / "face-profile"
CAL = (160, 184)


def identity():
    files = (
        "pores/FACE-PROFILE-PILOT.md",
        "pores/face_profile.cu",
        "scripts/pore_face_gpu.py",
        "scripts/pore_face_profile.py",
        "scripts/pore_face_profile_score.py",
        "scripts/pore_transfer.py",
        "scripts/pore_section_score.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_axial_score.py",
        "scripts/pore_object_catalog.py",
        "pores/local_model.py",
        "pores/pore_check.py",
        "pores/validate.py",
    )
    return dict(
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in files
        },
        parent=sha(BASE / "predictions.json"),
        compiled={k: sha(PTX / k) for k in ("face-profile.ptx", "pores.ptx", "cone-pores.exe")},
    )


def raw_slab(cache, i, sl):
    if i not in DEV or sl not in (CAL, Z):
        raise ValueError("Unregistered raw region")
    from local_model import context_stack

    path, r = verified(cache, i, "raw", False)
    parent = read(BASE / "predictions.json")["specimens"][str(i)]
    assert r["sha256"] == parent["raw"]
    a, b = sl
    raw = context_stack(path, i, a - 4, b + 4, 5)
    assert raw.shape == (b - a + 10, 266, 266)
    return np.ascontiguousarray(raw), parent


def faces(m, eligible=None):
    for ax in range(3):
        for sign in (-1, 1):
            x = np.moveaxis(m, ax, 0)
            f = np.zeros_like(x)
            if sign == 1:
                f[:-1] = x[:-1] & ~x[1:]
            else:
                f[1:] = x[1:] & ~x[:-1]
            f = np.moveaxis(f, 0, ax)
            if eligible is not None:
                f &= eligible
            pts = np.argwhere(f)
            if len(pts):
                yield ax, sign, pts


def scalar(raw, pt, ax, sign):
    t1, t2 = [k for k in range(3) if k != ax]
    res = []
    for shift in (-1, 0, 1):
        line = []
        b = []
        for k in range(-3, 5):
            q = pt.copy() + 5
            q[ax] += (k + shift) * sign
            line.append(int(raw[tuple(q)]))
            v = 0
            for x in (-1, 0, 1):
                for y in (-1, 0, 1):
                    t = q.copy()
                    t[t1] += x
                    t[t2] += y
                    v += int(raw[tuple(t)])
            b.append(v)
        res.append([255 * (v - min(a)) // max(1, max(a) - min(a)) for a in (line, b) for v in a])
    return res


def features(raw, pts, ax, sign):
    if (
        raw.dtype != np.uint16
        or raw.ndim != 3
        or raw.size >= 2**31
        or ax not in (0, 1, 2)
        or sign not in (-1, 1)
        or pts.ndim != 2
        or pts.shape[1] != 3
        or not len(pts)
        or pts.min() < 0
        or np.any(pts.max(0) >= np.array(raw.shape) - 10)
    ):
        raise ValueError("Invalid raw geometry")
    z, y, x = raw.shape
    steps = (y * x, x, 1)
    u, v = [k for k in range(3) if k != ax]
    s = sign * steps[ax]
    p = np.ravel_multi_index((pts + 5).T, raw.shape).astype("i4")
    gpu = native(raw, p, s, steps[u], steps[v])
    r = raw.ravel()
    idx = p[:, None] + np.arange(-4, 6, dtype="i4") * s
    line = r[idx].astype("i4")
    plane = np.zeros_like(line)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            plane += r[idx + a * steps[u] + b * steps[v]]
    for j in range(3):
        for k, a in enumerate((line, plane)):
            c = a[:, j : j + 8]
            lo = c.min(1)[:, None]
            hi = c.max(1)[:, None]
            assert np.array_equal(
                (c - lo) * 255 // np.maximum(hi - lo, 1), gpu[:, j, k * 8 : k * 8 + 8]
            )
    assert np.array_equal(scalar(raw, pts[len(pts) // 2], ax, sign), gpu[len(pts) // 2])
    return gpu


def batches(raw, m, eligible=None):
    for ax, sign, pts in faces(m, eligible):
        for a in range(0, len(pts), 32768):
            q = pts[a : a + 32768]
            yield ax, sign, q, features(raw, q, ax, sign)


def fit(cache):
    G = np.zeros((16, 16))
    h = np.zeros(16)
    rec = {}
    diff = []
    for i in DEV:
        raw, r = raw_slab(cache, i, CAL)
        f, s = verified(cache, i, "segmented", False)
        assert s["sha256"] == r["segmented"]
        m = stack(f, i, "segmented", *CAL)
        assert digest(m) == r["cal_mask"]
        xx = np.zeros((16, 16), "i8")
        xy = np.zeros(16, "i8")
        n = 0
        parts = []
        for _, _, _, feat in batches(raw, m):
            for j in (0, 2):
                d = feat[:, 1].astype("i8") - feat[:, j]
                xx += d.T @ d
                xy += d.sum(0)
                n += len(d)
                parts.append(d.astype("i2"))
        assert n > 0
        G += xx / (255**2 * n * len(DEV))
        h += xy / (255 * n * len(DEV))
        diff.append(np.concatenate(parts))
        rec[str(i)] = dict(
            raw=r["raw"], calibration_mask=digest(m), pairs=n, gram=xx.tolist(), rhs=xy.tolist()
        )
    w = np.rint(np.linalg.solve(G + 0.01 * np.eye(16), h) * 65536).astype("i8")
    assert w.any() and np.max(abs(w)) < 2**30
    for i, d in zip(DEV, diff, strict=True):
        v = d @ w
        rec[str(i)].update(prefer_true=int((v > 0).sum()), ties=int((v == 0).sum()))
    return w, rec


def choices(feat, w):
    v = feat.astype("i8") @ w
    # Argmax on the priority order stay, inward, outward.
    j = np.array([1, 0, 2])[np.argmax(v[:, [1, 0, 2]], axis=1)]
    return j - 1, v[np.arange(len(v)), j] - v[:, 1]


def select(raw, m, w):
    a, sz, whole, boxes = label(m)
    eligible = whole & (sz >= 64)
    out = m.copy()
    expected = sz.copy()
    axmoves = np.zeros((len(sz), 3), "i8")
    positions = np.zeros((len(sz), 3), "i8")
    proposal = {}
    nf = 0
    for ax, sign, pts, feat in batches(raw, m, eligible[a]):
        dr, gain = choices(feat, w)
        labs = a[tuple(pts.T)]
        np.add.at(axmoves[:, ax], labs, dr)
        np.add.at(positions, (labs, dr + 1), 1)
        nf += len(pts)
        for j in np.unique(labs[dr != 0]):
            ix = (labs == j) & (dr != 0)
            q = pts[ix].copy()
            q[:, ax] += sign * (dr[ix] > 0)
            proposal.setdefault(int(j), []).append(
                (np.ravel_multi_index(q.T, a.shape), dr[ix], gain[ix])
            )
    rows = {}
    for j in map(int, np.flatnonzero(eligible)):
        size = int(sz[j])
        delta = (int(axmoves[j].sum()) + 1) // 3
        target = size + delta
        box = tuple(
            slice(max(0, s.start - 3), min(n, s.stop + 3))
            for s, n in zip(boxes[j - 1], m.shape, strict=True)
        )
        loc = a[box]
        own = loc == j
        new = own.copy()
        nc = 0
        reason = "equal"
        ids = np.array([], dtype="<i8")
        if delta:
            v, d, g = [np.concatenate(z) for z in zip(*proposal[j], strict=True)]
            use = d == np.sign(delta)
            uq, inv, cnt = np.unique(v[use], return_inverse=True, return_counts=True)
            gs = np.zeros(len(uq), "i8")
            np.add.at(gs, inv, g[use])
            q = np.array(np.unravel_index(uq, m.shape)).T
            pc = q - np.array([s.start for s in box])
            lx = tuple(pc.T)
            if delta > 0:
                assert not own[lx].any()
                bad = ndi.binary_dilation((loc != 0) & ~own, FACE, iterations=2)
                keep = ~bad[lx] & np.all((q > 0) & (q < np.array(m.shape) - 1), axis=1)
            else:
                assert own[lx].all()
                keep = np.ones(len(q), bool)
            ids = uq[keep]
            cnt = cnt[keep]
            gs = gs[keep]
            nc = len(ids)
            order = np.lexsort((ids, -gs, -cnt))
            ids = ids[order[: abs(delta)]]
            reason = "edited"
            if target < 64:
                reason = "below 64"
            elif nc < abs(delta):
                reason = "insufficient sites"
            else:
                lq = np.array(np.unravel_index(ids, m.shape)).T - np.array([s.start for s in box])
                ix = tuple(lq.T)
                new[ix] = delta > 0
                if ndi.label(new, FACE)[1] != 1:
                    reason = "disconnected"
                else:
                    assert np.array_equal(out[box][ix], own[ix])
                    out[box][ix] = new[ix]
                    expected[j] = target
            if reason != "edited":
                ids = ids[:0]
        rows[str(j)] = dict(
            detector=size,
            axis_delta=axmoves[j].tolist(),
            in_stay_out=positions[j].tolist(),
            target=target,
            actual=int(expected[j]),
            candidates=nc,
            changed=len(ids),
            reason=reason,
            indices=digest(np.sort(ids).astype("<i8")),
        )
    b, v, ww, _ = label(out)
    both = m & out
    k = np.unique(a[both].astype("i8") * len(v) + b[both])
    p, q = np.divmod(k, len(v))
    assert (
        np.array_equal(p, np.arange(1, len(sz)))
        and np.array_equal(np.sort(q), np.arange(1, len(v)))
        and np.array_equal(v[q], expected[p])
        and np.array_equal(ww[q], whole[p])
    )
    return out, dict(
        objects=rows,
        components=len(k),
        oriented_faces=nf,
        feature_bytes_checked=nf * 48,
        reasons=dict(Counter(x["reason"] for x in rows.values())),
    )


def run(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    ident = identity()
    w, cal = fit(cache)
    f = out / "fit.json"
    write(f, dict(weights=w.tolist(), calibration=cal))
    spec = {}
    for i in DEV:
        raw, r = raw_slab(cache, i, Z)
        m = read_mask(BASE, i, "curvature", r["masks"]["curvature"])
        n, g = select(raw, m, w)
        npz = out / f"sample{i}-face.npz"
        np.savez_compressed(npz, mask=n)
        ledger = out / f"sample{i}-choices.json"
        write(ledger, g)
        spec[str(i)] = dict(
            raw=r["raw"],
            segmented=r["segmented"],
            detector=digest(m),
            ledger=sha(ledger),
            masks=dict(face=dict(file=sha(npz), mask=digest(n))),
        )
        print(i, g["oriented_faces"], g["reasons"], flush=True)
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            fit=sha(f),
            origin_xyz=[360, 376, Z[0]],
            shape_zyx=list(SHAPE),
            specimens=spec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("cache", "out"):
        ap.add_argument("--" + key, type=Path, required=True)
    args = ap.parse_args()
    run(args.cache, args.out)
