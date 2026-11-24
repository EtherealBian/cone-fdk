"""Development-only signed native-count integration. No transfer reference input."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, ROOT, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_mass_gpu import terms
from pore_small_seed import E as SEED
from pore_transfer import FACE, label
from scipy import ndimage as ndi

E = ROOT / "pores/evidence/mass"
ARMS = ("mass", "size", "boundary")


def identity():
    files = (
        "pores/MASS-PILOT.md",
        "scripts/pore_mass.py",
        "scripts/pore_mass_score.py",
        "scripts/pore_mass_gpu.py",
        "scripts/pore_mass.cu",
        "scripts/pore_axial_raw.py",
        "scripts/pore_axial_score.py",
        "scripts/pore_transfer.py",
        "pores/validate.py",
        "pores/pore_check.py",
    )
    return dict(
        mass_ptx=sha(ROOT / "build/pores/mass-terms.ptx"),
        native=sha(ROOT / "build/pores/cone-pores.exe"),
        files={
            p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in files
        },
        parents=[sha(p / "predictions.json") for p in (EVID, SEED)],
    )


def calibration(raw, mask):
    result = {}
    for name, phase in [("pore", mask), ("material", ~mask)]:
        deep = ndi.distance_transform_edt(np.pad(phase, 1))[1:-1, 1:-1, 1:-1] > 3
        v = raw[deep]
        result[name] = dict(n=v.size, median_x2=int(2 * np.median(v)) if len(v) else None)
    p, m = result.values()
    result["valid"] = min(p["n"], m["n"]) >= 64 and m["median_x2"] > p["median_x2"]
    return result


def estimate(raw, obj, other, pore2):
    d2 = np.rint(ndi.distance_transform_edt(~obj) ** 2).astype("i4")
    a, inside = raw[(d2 > 16) & (d2 <= 36)], raw[d2 <= 9]
    na, sa, ni, si, nf = terms(raw, obj, other)
    assert (na, sa, ni, si, nf) == (
        a.size,
        int(a.sum(dtype="i8")),
        inside.size,
        int(inside.sum(dtype="i8")),
        int(np.count_nonzero(other & (d2 <= 36))),
    )
    if nf:
        return dict(reason="neighbor", foreign_sites=nf)
    numerator, denominator = 2 * (ni * sa - na * si), 2 * sa - na * pore2
    assert numerator == sum(2 * (sa - na * int(x)) for x in inside)
    return dict(
        reason="ok" if denominator > 0 else "nonpositive_contrast",
        n_annulus=na,
        sum_annulus=sa,
        n_integrated=ni,
        sum_integrated=si,
        numerator=numerator,
        denominator=denominator,
    )


def boundary(obj, raw, numerator, denominator):
    n = int(obj.sum())
    target = (2 * numerator + denominator) // (2 * denominator)
    delta = max(-(n // 4), min(n // 4, target - n))
    new = obj.copy()
    if delta:
        face = (
            ndi.binary_dilation(obj, FACE) & ~obj
            if delta > 0
            else obj & ~ndi.binary_erosion(obj, FACE)
        )
        ix = np.flatnonzero(face)
        order = np.lexsort((ix, (1 if delta > 0 else -1) * raw.flat[ix].astype("i4")))
        done = 0
        for at in ix[order]:
            new.flat[at] = delta > 0
            if delta < 0 and ndi.label(new, FACE)[1] != 1:
                new.flat[at] = True
                continue
            done += 1
            if done == abs(delta):
                break
    return new, delta


def expanded(box, n, dims):
    return tuple(
        slice(max(0, s.start - n), min(dim, s.stop + n)) for s, dim in zip(box, dims, strict=True)
    )


def make(raw, base, gray, cal):
    lab, sizes, whole, boxes = label(base)
    gl, gs, gw, gb = label(gray)
    assert np.all(gray[base]) and len(boxes) == len(gb)
    out = {k: base.copy() for k in ARMS}
    rows = []
    proposals = {}
    seen = set()
    for j, box in enumerate(boxes, 1):
        obj = lab[box] == j
        anchor = tuple(np.argwhere(obj)[0] + [s.start for s in box])
        g = int(gl[anchor])
        assert g not in seen and gw[g] == whole[j]
        seen.add(g)
        n = int(sizes[j])
        sid = int(np.ravel_multi_index(anchor, lab.shape)) + 1
        if not whole[j] or not 8 <= n <= 512:
            assert n == gs[g]
            continue
        bb = expanded(box, 6, lab.shape)
        local = lab[bb]
        obj = local == j
        if not cal["valid"]:
            est = dict(reason="calibration")
        elif any(s.start < 6 or s.stop + 6 > dim for s, dim in zip(box, lab.shape, strict=True)):
            est = dict(reason="edge")
        else:
            est = estimate(raw[bb], obj, (local != 0) & ~obj, cal["pore"]["median_x2"])
        ok = est["reason"] == "ok"
        row = dict(
            seed=sid,
            original=n,
            prior_gray=int(gs[g]),
            label=j,
            estimate=est,
            mass=False,
            size=False,
            requested_delta=0,
            applied_delta=0,
        )
        rows.append(row)
        if n < 64 and gs[g] >= 64:
            gg = gb[g - 1]
            proposals[sid] = (gg, gl[gg] == g)
            row["mass"] = ok and est["numerator"] >= 64 * est["denominator"]
        if n >= 64 and ok:
            assert gs[g] == n
            new, delta = boundary(obj, raw[bb], est["numerator"], est["denominator"])
            row.update(requested_delta=delta, applied_delta=int(new.sum()) - n)
            out["boundary"][bb] ^= new ^ obj
    ranked = sorted(
        (r for r in rows if r["seed"] in proposals), key=lambda r: (-r["original"], r["seed"])
    )
    for r in ranked[: sum(r["mass"] for r in ranked)]:
        r["size"] = True
    for r in ranked:
        bb, proposal = proposals[r["seed"]]
        for k in ARMS:
            if r["size" if k == "size" else "mass"]:
                out[k][bb] |= proposal
    for k, m in out.items():
        nl, ns, nw, nb = label(m)
        assert len(nb) == len(boxes)
        keys = np.unique(np.stack([lab[base & m], nl[base & m]]), axis=1)
        assert keys.shape == (2, len(boxes)) and np.unique(keys[1]).size == len(nb)
        assert np.array_equal(nw[keys[1]], whole[keys[0]])
        expected = sizes.copy()
        for r in rows:
            j = r["label"]
            if r["size" if k == "size" else "mass"]:
                expected[j] = r["prior_gray"]
            elif k == "boundary":
                expected[j] += r["applied_delta"]
        assert np.array_equal(ns[keys[1]], expected[keys[0]])
    return out, rows, len(boxes) - len(rows)


def predict(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    prior, parent = [
        json.loads((p / "predictions.json").read_bytes())["specimens"] for p in (SEED, EVID)
    ]
    specimens = {}
    for i in DEV:
        pre, par = prior[str(i)], parent[str(i)]
        raw, rs = verified(cache, i, "raw", False)
        seg, ss = verified(cache, i, "segmented", False)
        assert (
            rs["sha256"] == pre["raw"] == par["raw"]
            and ss["sha256"] == pre["segmented"] == par["segmented"]
        )
        cr = stack(raw, i, "raw", 160, 184)
        ct = stack(seg, i, "segmented", 160, 184)
        cal = calibration(cr, ct)
        cal.update(z=[160, 184], raw_digest=digest(cr), label_digest=digest(ct))
        pix = stack(raw, i, "raw", *Z)
        base = read_mask(EVID, i, "census", par["masks"]["census"])
        assert digest(base) == pre["census"]
        gray = read_mask(SEED, i, "gray", pre["masks"]["gray"])
        masks, rows, untouched = make(pix, base, gray, cal)
        f = out / f"sample{i}-estimates.json"
        write(f, rows)
        rec = dict(
            calibration=cal,
            raw=rs["sha256"],
            segmented=ss["sha256"],
            raw_roi=digest(pix),
            census=digest(base),
            gray=digest(gray),
            estimates=sha(f),
            outside_cohort=untouched,
            masks={},
        )
        for k, m in masks.items():
            path = out / f"sample{i}-{k}.npz"
            with path.open("xb") as fp:
                np.savez_compressed(fp, mask=m)
            rec["masks"][k] = {"mask": digest(m), "file": sha(path)}
        specimens[str(i)] = rec
        print(i, cal, len(rows), "promoted", sum(r["mass"] for r in rows), flush=True)
    ident = identity()
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            specimens=specimens,
            origin_xyz=[360, 376, Z[0]],
            shape_zyx=list(SHAPE),
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Do not run with -O")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    predict(a.cache, a.out)
