"""Fit a signed XY section-area residual, then choose one whole sealed endpoint per pore."""

import argparse
import hashlib
from collections import Counter
from fractions import Fraction
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
    features,
    sha,
    stack,
    token,
    verified,
    write,
)
from pore_chord_fraction import E as CHORD
from pore_chord_fraction import biject, sources
from pore_curvature_fields import field
from pore_face_profile import BASE, CAL
from pore_face_profile import E as FACE_E
from pore_local_profile import library_score, load_model, profiles
from pore_profile_gpu import gpu_score, pack
from pore_region import EV, read
from pore_transfer import label
from pore_xy_gpu import rings
from scipy import ndimage as ndi

E = EV / "section-gate"
XY = ndi.generate_binary_structure(2, 1)


def identity():
    old = read(CHORD / "predictions.json")["identity"]
    names = set(old["files"]) | {
        "pores/SECTION-GATE-PILOT.md",
        "pores/section_gate.cu",
        "scripts/pore_section_gate.py",
        "scripts/pore_section_gate_score.py",
        "scripts/pore_xy_gpu.py",
        "pores/local_model.py",
        "scripts/pore_region.py",
    }
    return dict(
        files={
            f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for f in sorted(names)
        },
        parents={q.name: sha(q / "predictions.json") for q in (BASE, FACE_E, CHORD)},
        compiled={k: sha(PTX / k) for k in sorted(set(old["ptx"]) | {"section-gate.ptx"})},
    )


def load_source(cache, i, sl, kind):
    if (
        i not in DEV
        or sl not in (Z, CAL)
        or kind not in ("raw", "segmented")
        or (kind == "segmented" and sl != CAL)
    ):
        raise ValueError("Unregistered prediction input")
    path, r = verified(cache, i, kind, False)
    assert r["sha256"] == read(BASE / "predictions.json")["specimens"][str(i)][kind]
    return path, stack(path, i, kind, *sl)


def detector(path, i):
    if i not in DEV:
        raise ValueError("Unregistered specimen")
    seal = read(BASE / "predictions.json")
    f = BASE / "curvature.npz"
    assert sha(f) == seal["fits"]["curvature"]["weights"]
    bias, nodes = load_model(f)
    arr = pack(nodes, 61)
    m = np.empty((24, 256, 256), bool)
    with Denoise(PTX / "search-support.ptx") as den:
        for z in range(*CAL, 4):
            x = features(path, i, z, z + 4, den)
            c = field(path, i, z, z + 4)
            for j in range(4):
                v = np.concatenate((profiles(x[j]), c[j].reshape(-1, 15)), axis=1)
                q = gpu_score(v, arr, bias)
                assert np.array_equal(q > 0, library_score(v, nodes, bias) > 0)
                m[z - CAL[0] + j] = (q > 0).reshape(256, 256)
    f = CHORD / "fit.json"
    assert (
        sha(f) == read(CHORD / "predictions.json")["fit"]
        and digest(m) == read(f)["specimens"][str(i)]["mask"]
    )
    return m


def planar(m):
    a, n = ndi.label(m, XY)
    sizes = np.bincount(a.ravel(), minlength=n + 1)
    boxes = ndi.find_objects(a)
    valid = {
        j
        for j, b in enumerate(boxes, 1)
        if 4 <= sizes[j] <= 2048
        and all(t.start >= 3 and t.stop <= s - 3 for t, s in zip(b, a.shape, strict=True))
    }
    return a, sizes, boxes, valid


def match(a, b):
    p, v, _, ok = a
    r, u, _, good = b
    idx = (p > 0) & (r > 0)
    keys, hits = np.unique(p[idx].astype("i8") * len(u) + r[idx], return_counts=True)
    edges = []
    for k, h in zip(keys, hits, strict=True):
        j, t = divmod(int(k), len(u))
        union = int(v[j] + u[t] - h)
        if j in ok and t in good and 2 * h >= union:
            edges.append((-Fraction(int(h), union), j, t))
    used = set()
    out = {}
    for _, j, t in sorted(edges):
        if j not in out and t not in used:
            out[j] = t
            used.add(t)
    return out


def cpu_rings(m):
    own = m == 1
    near = ndi.binary_dilation(own, XY)
    far = ndi.binary_dilation(own, XY, iterations=3)
    deg = ndi.convolve(own.astype("u1"), XY.astype("u1"), mode="constant", cval=0) - own
    return np.where(
        own, 4 - deg, np.where(m == 2, 7, np.where(near, 5, np.where(far, 6, 7)))
    ).astype("u1")


def descriptor(raw, c, scale):
    bd = (c >= 1) & (c <= 4)
    own = c <= 4
    outer = c == 5
    far = c == 6
    if not outer.any() or not far.any():
        return None
    p = int(c[bd].sum())
    area = int(own.sum())
    core = c == 0
    lo = float(np.median(raw[core if core.any() else own]))
    high = float(np.median(raw[far]))
    k = max(1, high - lo)
    b, o = [np.clip((raw[m].astype("f8") - lo) / k, -2, 3) for m in (bd, outer)]
    return p, [
        1,
        area**-0.5,
        4 * np.pi * area / p**2,
        b.mean(),
        o.mean(),
        o.mean() - b.mean(),
        b.std(),
        o.std(),
        (b < 0.5).mean(),
        (o < 0.5).mean(),
        min(4, k * scale / 64),
    ]


def sections(raw, m, scale, truth=None, eligible=None):
    lab, sizes, whole, _ = label(m)
    result = []
    for z in range(len(m)):
        a = planar(m[z])
        ids, areas, boxes, valid = a
        if truth is not None:
            t = planar(truth[z])
            matched = match(a, t)
            js = sorted(matched)
        else:
            js = range(1, len(areas))
        rec = []
        for j in js:
            b = boxes[j - 1]
            own = ids[b] == j
            pid = int(lab[z][b][own][0])
            assert np.all(lab[z][b][own] == pid)
            if eligible is not None and pid not in eligible:
                continue
            row = dict(
                z=z, section=j, parent=pid, area=int(areas[j]), clipped=not bool(whole[pid])
            )
            if truth is not None:
                rid = matched[j]
                row.update(ref=rid, ref_area=int(t[1][rid]))
            result.append(row)
            if j not in valid:
                continue
            box = tuple(slice(s.start - 3, s.stop + 3) for s in b)
            crop = ids[box]
            owner = np.where(crop == j, 1, np.where(crop != 0, 2, 0)).astype("u1")
            rec.append((row, owner, raw[z][box]))
        for start in range(0, len(rec), 32):
            batch = rec[start : start + 32]
            h, w = np.max([v.shape for _, v, _ in batch], axis=0)
            atlas = np.full((len(batch), h, w), 2, "u1")
            for idx, (_, v, _) in enumerate(batch):
                atlas[idx, : v.shape[0], : v.shape[1]] = v
            out = rings(atlas)
            for tile, (row, v, img) in zip(out, batch, strict=True):
                c = tile[: v.shape[0], : v.shape[1]]
                assert np.array_equal(c, cpu_rings(v))
                q = descriptor(img, c, scale)
                if q is not None:
                    row["perimeter"], row["x"] = q
    if eligible is not None:
        tally = Counter()
        for d in result:
            tally[d["parent"]] += d["area"]
        assert all(tally[j] == sizes[j] for j in eligible)
    return result


def solve(pop):
    gram = np.diag([0] + [0.02] * 10)
    rhs = np.zeros(11)
    for rows in pop.values():
        good = [t for t in rows if "x" in t]
        parents = Counter(t["parent"] for t in good)
        if not parents:
            raise ValueError("No contributing calibration parent")
        for t in good:
            x = np.array(t["x"])
            y = (t["ref_area"] - t["area"]) / t["perimeter"]
            w = 1 / (len(pop) * len(parents) * parents[t["parent"]])
            gram += w * np.outer(x, x)
            rhs += w * x * y
    return np.rint(np.linalg.solve(gram, rhs) * 1e6).astype("i8")


def correction(row, w):
    return (
        0
        if "x" not in row
        else int(np.rint(1024 * row["perimeter"] * np.clip(np.dot(row["x"], w) / 1e6, -1, 1)))
    )


def prefer(delta, v, u):
    return abs(delta - 1024 * (u - v)) < abs(delta)


def choose(m, end, parents, rows, w):
    a, b = label(m), label(end)
    mapping = biject(a, b)
    change = {}
    deltas = Counter()
    out = m.copy()
    for row in rows:
        row["correction_1024"] = correction(row, w)
        deltas[row["parent"]] += row["correction_1024"]
    expected = {}
    for j, k in mapping.items():
        v, u = map(int, (a[1][j], b[1][k]))
        use = False
        if str(j) in parents:
            assert (v, u) == (
                parents[str(j)]["detector"],
                parents[str(j)]["actual"],
            ) and j in deltas
            use = prefer(deltas[j], v, u)
            change[str(j)] = dict(
                detector=v,
                full=u,
                delta_1024=deltas[j],
                endpoint="face" if use else "curvature",
                actual=u if use else v,
            )
        else:
            assert v == u
        box = tuple(
            slice(min(a1.start, b1.start), max(a1.stop, b1.stop))
            for a1, b1 in zip(a[3][j - 1], b[3][k - 1], strict=True)
        )
        old = a[0][box] == j
        r = b[0][box] == k
        if str(j) not in parents:
            assert np.array_equal(old, r)
        if use:
            out[box][old != r] = r[old != r]
        expected[j] = (box, k, use)
    c = label(out)
    new = biject(a, c)
    for j, (box, k, use) in expected.items():
        mask = (b[0][box] == k) if use else (a[0][box] == j)
        assert np.array_equal(c[0][box] == new[j], mask) and c[1][new[j]] == mask.sum()
    return out, change


def run(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    ident = identity()
    pop = {}
    rec = {}
    for i in DEV:
        path, raw = load_source(cache, i, CAL, "raw")
        _, truth = load_source(cache, i, CAL, "segmented")
        m = detector(path, i)
        assert digest(truth) == read(BASE / "predictions.json")["specimens"][str(i)]["cal_mask"]
        rows = sections(raw, m, SETTINGS[i][0], truth=truth)
        pop[str(i)] = rows
        good = [r for r in rows if "x" in r]
        rec[str(i)] = dict(
            mask=digest(m),
            reference=digest(truth),
            raw=digest(raw),
            sections=len(good),
            parents=len({r["parent"] for r in good}),
            clipped_parents=len({r["parent"] for r in good if r["clipped"]}),
        )
        print("cal", i, rec[str(i)], flush=True)
    w = solve(pop)
    for rows in pop.values():
        for row in rows:
            row["correction_1024"] = correction(row, w)
    f = out / "fit.json"
    write(f, dict(weights=w.tolist(), calibration=rec, rows=pop))
    spec = {}
    for i in DEV:
        (m, e), g, src = sources(i)
        _, raw = load_source(cache, i, Z, "raw")
        elig = set(map(int, g["objects"]))
        rows = sections(raw, m, SETTINGS[i][0], eligible=elig)
        res, choices = choose(m, e, g["objects"], rows, w)
        ledger = out / f"sample{i}-choices.json"
        write(ledger, dict(objects=choices, sections=rows))
        p = out / f"sample{i}-gate.npz"
        np.savez_compressed(p, mask=res)
        spec[str(i)] = dict(
            raw=src[0],
            segmented=src[1],
            parents=[digest(m), digest(e)],
            ledger=sha(ledger),
            masks=dict(gate=dict(file=sha(p), mask=digest(res))),
        )
        print(i, Counter(d["endpoint"] for d in choices.values()), flush=True)
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            shape_zyx=list(SHAPE),
            origin_xyz=[360, 376, Z[0]],
            fit=sha(f),
            specimens=spec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
