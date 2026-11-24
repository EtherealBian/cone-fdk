"""One source-calibrated fraction of an oriented face estimate; no target IDs at prediction."""

import argparse
import hashlib
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from pore_axial_raw import (
    DEV,
    PTX,
    ROOT,
    SHAPE,
    Denoise,
    Z,
    digest,
    features,
    sha,
    stack,
    token,
    verified,
    write,
)
from pore_axial_score import read_mask
from pore_curvature_fields import field
from pore_face_profile import BASE, CAL, raw_slab, select
from pore_face_profile import E as FACE_E
from pore_local_profile import library_score, load_model, profiles
from pore_profile_gpu import gpu_score, pack
from pore_region import EV, read
from pore_shell_gpu import fields
from pore_transfer import FACE, label, pairs
from scipy import ndimage as ndi

E = EV / "chord-fraction"


def identity():
    files = (
        "pores/CHORD-FRACTION-PILOT.md",
        "scripts/pore_chord_fraction.py",
        "scripts/pore_chord_fraction_score.py",
        "scripts/pore_fraction_bound.py",
        "scripts/pore_face_profile.py",
        "scripts/pore_face_gpu.py",
        "scripts/pore_shell_gpu.py",
        "scripts/pore_axial_raw.py",
        "scripts/pore_curvature_fields.py",
        "scripts/pore_local_profile.py",
        "scripts/pore_profile_gpu.py",
        "scripts/pore_transfer.py",
        "scripts/pore_axial_score.py",
        "scripts/pore_section_score.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_object_catalog.py",
        "pores/pore_check.py",
        "pores/validate.py",
    )
    return dict(
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in files
        },
        parents={q.name: sha(q / "predictions.json") for q in (BASE, FACE_E)},
        ptx={
            k: sha(PTX / k)
            for k in (
                "shell-volume.ptx",
                "face-profile.ptx",
                "volume-curvature.ptx",
                "local-profile.ptx",
                "search-support.ptx",
                "pores.ptx",
                "cone-pores.exe",
            )
        },
    )


def rounded(v, u, t):
    q = (u - v) * t + F(1, 2)
    return v + q.numerator // q.denominator


def fraction(samples):
    opts = []
    for rows in samples:
        for v, u, r in rows:
            if u != v:
                opts.append((F(r - v, u - v), F(abs(u - v), r * len(rows))))
    if not opts:
        return F(0)
    W = sum(w for _, w in opts)
    cum = F(0)
    for pos, w in sorted(opts):
        cum += w
        if 2 * cum >= W:
            return min(F(1), max(F(0), pos))
    raise AssertionError("invalid weights")


def calibrate(cache):
    d = read(BASE / "predictions.json")
    fp = BASE / "curvature.npz"
    assert sha(fp) == d["fits"]["curvature"]["weights"]
    base, nodes = load_model(fp)
    arr = pack(nodes, 61)
    f = FACE_E / "fit.json"
    assert sha(f) == read(FACE_E / "predictions.json")["fit"]
    w = np.array(read(f)["weights"], "i8")
    sample = {}
    allpairs = []
    for i in DEV:
        raw, src = raw_slab(cache, i, CAL)
        path, s = verified(cache, i, "raw", False)
        assert s["sha256"] == src["raw"]
        m = np.empty((24, 256, 256), bool)
        hp = hashlib.sha256()
        hc = hashlib.sha256()
        md = 0.0
        with Denoise(PTX / "search-support.ptx") as den:
            for a in range(160, 184, 4):
                x = features(path, i, a, a + 4, den)
                c = field(path, i, a, a + 4)
                for j in range(4):
                    z = a - 160 + j
                    v = profiles(x[j])
                    cf = c[j]
                    hp.update(
                        np.ascontiguousarray(
                            v.reshape(256, 256, 46)[z % 2 :: 2, (z // 2) % 2 :: 2]
                        ).view("u1")
                    )
                    hc.update(np.ascontiguousarray(cf[z % 2 :: 2, (z // 2) % 2 :: 2]).view("u1"))
                    full = np.concatenate((v, cf.reshape(-1, 15)), axis=1)
                    native = gpu_score(full, arr, base)
                    cpu = library_score(full, nodes, base)
                    assert np.array_equal(cpu > 0, native > 0)
                    md = max(md, float(np.max(abs(cpu - native))))
                    m[z] = (native > 0).reshape(256, 256)
        assert hp.hexdigest() == src["profiles"] and hc.hexdigest() == src["curvature_cal"]
        full, g = select(raw, m, w)
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == src["segmented"]
        truth = stack(seg, i, "segmented", *CAL)
        assert digest(truth) == src["cal_mask"]
        det = label(m)
        tx = label(truth)
        rr = {}
        rows = []
        for rid, (pid, iou) in sorted(pairs(det, tx, 64).items()):
            v, u, r = int(det[1][pid]), g["objects"][str(pid)]["actual"], int(tx[1][rid])
            rr[str(rid)] = dict(detector_label=int(pid), detector=v, full=u, reference=r, iou=iou)
            rows.append((v, u, r))
        allpairs.append(rows)
        sample[str(i)] = dict(
            rows=rr,
            raw=src["raw"],
            mask=digest(m),
            full_mask=digest(full),
            cal_reference=digest(truth),
            max_cpu_cuda_score_difference=md,
            profiles=hp.hexdigest(),
            curvature_cal=hc.hexdigest(),
            directional_pairs=sum(u != v for v, u, r in rows),
        )
        print("cal", i, len(rows), flush=True)
    t = fraction(allpairs)

    def loss(q):
        vals = [
            sum(abs(F(v) + (u - v) * q - r) / r for v, u, r in rows) / len(rows)
            for rows in allpairs
            if rows
        ]
        return str(sum(vals) / len(vals)) if vals else None

    return t, dict(
        fraction=str(t),
        specimens=sample,
        mean_relative_l1={str(v): loss(v) for v in (F(0), t, F(1))},
    )


def biject(a, b):
    x, y = a[0], b[0]
    n = len(b[1])
    over = (x > 0) & (y > 0)
    keys = np.unique(x[over].astype("i8") * n + y[over])
    p, q = np.divmod(keys, n)
    assert (
        np.array_equal(p, np.arange(1, len(a[1])))
        and np.array_equal(np.sort(q), np.arange(1, n))
        and np.array_equal(a[2][p], b[2][q])
    )
    return {int(i): int(j) for i, j in zip(p, q, strict=True)}


def project(m, endpoint, ledger, t):
    assert F(0) <= t <= F(1)
    a, b = label(m), label(endpoint)
    mapping = biject(a, b)
    out = m.copy()
    rows = {}
    expected = a[1].copy()
    assert set(map(int, ledger["objects"])) == set(np.flatnonzero(a[2] & (a[1] >= 64)))
    for j, k in mapping.items():
        if str(j) not in ledger["objects"]:
            assert int(a[1][j]) == int(b[1][k]) and np.array_equal(
                a[0][a[3][j - 1]] == j, b[0][a[3][j - 1]] == k
            )
            continue
        v, u = int(a[1][j]), int(b[1][k])
        parent = ledger["objects"][str(j)]
        assert (v, u) == (parent["detector"], parent["actual"])
        want = rounded(v, u, t)
        box = tuple(
            slice(min(p.start, q.start), max(p.stop, q.stop))
            for p, q in zip(a[3][j - 1], b[3][k - 1], strict=True)
        )
        own = a[0][box] == j
        end = b[0][box] == k
        diff = np.flatnonzero(own != end)
        assert len(diff) == abs(u - v)
        dv = want - v
        sel = diff[:0]
        why = "equal"
        if dv:
            degree = (fields(own.astype("u1"), np.zeros(own.shape, "u1")) >> 3) & 7
            cpu = ndi.convolve(own.astype("u1"), FACE.astype("u1"), mode="constant", cval=0) - own
            assert np.array_equal(degree, cpu)
            order = np.lexsort((diff, (-1 if dv > 0 else 1) * degree.ravel()[diff].astype("i2")))
            sel = diff[order[: abs(dv)]]
            x = own.copy()
            x.ravel()[sel] = dv > 0
            if ndi.label(x, FACE)[1] != 1:
                why = "disconnected"
                sel = sel[:0]
            else:
                why = "edited"
                ix = np.unravel_index(sel, own.shape)
                out[box][ix] = dv > 0
                expected[j] = want
        rows[str(j)] = dict(
            detector=v, full=u, target=want, actual=int(expected[j]), reason=why, changed=len(sel)
        )
    c = label(out)
    map2 = biject(a, c)
    assert all(c[1][q] == expected[p] for p, q in map2.items())
    return out, rows


def sources(i):
    if i not in DEV:
        raise ValueError("Unknown specimen")
    ms = []
    rs = []
    for f, k in ((BASE, "curvature"), (FACE_E, "face")):
        d = read(f / "predictions.json")
        assert d["shape_zyx"] == list(SHAPE) and d["origin_xyz"] == [360, 376, 256]
        r = d["specimens"][str(i)]
        rs.append((r["raw"], r["segmented"]))
        ms.append(read_mask(f, i, k, r["masks"][k]))
    assert rs[0] == rs[1]
    p = FACE_E / f"sample{i}-choices.json"
    assert sha(p) == r["ledger"]
    return ms, read(p), rs[0]


def run(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    ident = identity()
    t, cal = calibrate(cache)
    fit = out / "fit.json"
    write(fit, cal)
    spec = {}
    for i in DEV:
        (m, e), g, (raw, seg) = sources(i)
        res, lut = project(m, e, g, t)
        p = out / f"sample{i}-fraction.npz"
        np.savez_compressed(p, mask=res)
        f = out / f"sample{i}-choices.json"
        write(f, lut)
        spec[str(i)] = dict(
            raw=raw,
            segmented=seg,
            ledger=sha(f),
            parents=[digest(m), digest(e)],
            masks=dict(fraction=dict(file=sha(p), mask=digest(res))),
        )
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            shape_zyx=list(SHAPE),
            origin_xyz=[360, 376, Z[0]],
            fit=sha(fit),
            specimens=spec,
        ),
    )
    print("fraction", t, flush=True)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
