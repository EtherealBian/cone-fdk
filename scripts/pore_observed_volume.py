"""Count genuinely distinct observed-volume targets on original calibration slabs only."""

import argparse
import hashlib
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pore_axial_raw import DEV, ROOT, digest, sha, token, write
from pore_crossfit_gate import CAL, CAL_SHAPE, E, donors
from pore_region import read
from pore_section_gate import load_source
from pore_transfer import SPEC, label, native, reference

OUT = E.parent / "observed-volume-support"


def catalog(m):
    labs, sizes, _, boxes = label(m)
    out = {}
    for j, (z, y, x) in enumerate(boxes, 1):
        if (
            sizes[j] < 64
            or min(y.start, x.start) == 0
            or y.stop == m.shape[1]
            or x.stop == m.shape[2]
        ):
            continue
        out[j] = (int(sizes[j]), int(z.start == 0) + 2 * int(z.stop == m.shape[0]))
    return labs, out


def population(m, t):
    if m.shape != t.shape or m.ndim != 3 or m.dtype != bool or t.dtype != bool:
        raise ValueError("Expected two co-registered binary 3D masks")
    a, p = catalog(m)
    b, r = catalog(t)
    n = int(b.max()) + 1
    ix = (a > 0) & (b > 0)
    keys, hits = np.unique(a[ix].astype("i8") * n + b[ix], return_counts=True)
    edges = []
    for k, h in zip(keys, hits, strict=True):
        j, v = divmod(int(k), n)
        h = int(h)
        if j not in p or v not in r or p[j][1] != r[v][1]:
            continue
        u = p[j][0] + r[v][0] - h
        if 2 * h >= u:
            edges.append((-Fraction(h, u), j, v, h, u))
    rows = []
    used = set()
    found = set()
    for _, j, v, h, u in sorted(edges):
        if j in used or v in found:
            continue
        used.add(j)
        found.add(v)
        q, z = p[j]
        w = r[v][0]
        rows.append(
            dict(
                pred=j,
                ref=v,
                pred_voxels=q,
                ref_voxels=w,
                z_faces=z,
                intersection=h,
                union=u,
                residual=w - q,
            )
        )

    def group(rows):
        return dict(
            pairs=len(rows),
            under=sum(z["residual"] > 0 for z in rows),
            over=sum(z["residual"] < 0 for z in rows),
            tie=sum(z["residual"] == 0 for z in rows),
            outside=sum(100 * abs(z["residual"]) > 15 * z["ref_voxels"] for z in rows),
        )

    g = group(rows)
    return dict(
        predicted=len(p),
        reference=len(r),
        unmatched_pred={str(k): v for k, v in p.items() if k not in used},
        unmatched_ref={str(k): v for k, v in r.items() if k not in found},
        rows=sorted(rows, key=lambda d: d["pred"]),
        complete=group([z for z in rows if not z["z_faces"]]),
        z_cut=group([z for z in rows if z["z_faces"]]),
        all=g,
        support=g["pairs"] >= 5 and min(g["under"], g["over"]) >= 2,
    )


def components(out, i, m, tag):
    dest = out / f"sample{i}-{tag}.csv"
    s = (SPEC[i][2],) * 3
    with TemporaryDirectory() as tmp:
        src = Path(tmp) / "mask.f32"
        (~m).astype("<f4").tofile(src)
        st = native(src, dest, m.shape, s, 0, 6, 8, "none", 4)
        _, rows = reference(m, 6, s, dest, st)
        other = Path(tmp) / "second.csv"
        st2 = native(src, other, m.shape, s, 0, 6, 17, "none", 64)
        assert (st["slab_depth"], st2["slab_depth"]) == (
            8,
            17,
        ) and dest.read_bytes() == other.read_bytes()
    return dict(components=len(rows), table=sha(dest), mask=digest(m), slabs=[8, 17])


def run(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    replay = read(E / "prediction-replay.json")
    f = E / "fit.json"
    assert sha(f) == replay["fit.json"]
    folds = read(f)["calibration"]
    names = (
        "pores/OBSERVED-VOLUME-SUPPORT.md",
        "scripts/pore_observed_volume.py",
        "scripts/pore_transfer.py",
        "pores/pore_check.py",
        "pores/validate.py",
        "scripts/pore_section_gate.py",
    )
    ident = dict(
        source={
            n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in names
        },
        binaries={n: sha(ROOT / "build/pores" / n) for n in ("cone-pores.exe", "pores.ptx")},
        prediction_seal=sha(E / "prediction-replay.json"),
        fit=sha(f),
    )
    rec = {}
    for i in DEV:
        fold = folds[str(i)]
        p = E / f"sample{i}-oof.npz"
        assert (
            fold["held_out"] == i
            and fold["donors"] == list(donors(i))
            and sha(p) == fold["file"] == replay[p.name]
        )
        with np.load(p, allow_pickle=False) as z:
            m = z["mask"]
        path, t = load_source(cache, i, CAL, "segmented")
        assert (
            m.shape == t.shape == CAL_SHAPE
            and digest(m) == fold["mask"]
            and digest(t) == fold["reference"]
        )
        d = population(m, t)
        d.update(
            native={k: components(out, i, arr, k) for k, arr in [("oof", m), ("reference", t)]},
            archive=sha(path),
        )
        rec[str(i)] = d
        print(i, d["complete"], d["z_cut"], d["support"], flush=True)
    write(
        out / "support.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            calibration_z=list(CAL),
            specimens=rec,
            supported=all(v["support"] for v in rec.values()),
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    a = argparse.ArgumentParser(description=__doc__)
    for key in ("cache", "out"):
        a.add_argument("--" + key, type=Path, required=True)
    a = a.parse_args()
    run(a.cache, a.out)
