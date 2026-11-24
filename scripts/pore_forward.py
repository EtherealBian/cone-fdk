"""Calibration-only forward model of archived digital voids; spatial fit/check are separate."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_axial_raw import DEV, PTX, ROOT, digest, sha, token, verified, write
from pore_forward_gpu import native
from pore_mass import calibration
from pore_transfer import FACE
from scipy import ndimage as ndi
from validate import SPEC

E = ROOT / "pores/evidence/forward"
D = 98304
zz, yy, xx = np.ogrid[-2:3, -2:3, -2:3]
DIST = xx * xx + yy * yy + zz * zz
COUNTS = np.array([np.count_nonzero(DIST == j) for j in range(5)])
CONTROL = np.array([4096, 0, 0, 0, 0])


def identity():
    paths = [
        "pores/FORWARD-PILOT.md",
        "scripts/pore_forward.py",
        "scripts/pore_forward_gpu.py",
        "scripts/pore_forward.cu",
        "scripts/pore_mass.py",
        "pores/validate.py",
    ]
    return dict(
        files={
            p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in paths
        },
        ptx=sha(PTX / "pore-forward.ptx"),
    )


def part(path, i, kind, phase):
    if i not in DEV or phase not in ("fit", "check") or kind not in ("raw", "segmented"):
        raise ValueError("Unknown calibration request")
    lo = 360 + (0 if phase == "fit" else 136)
    st = f"set1sample{i}{kind}"
    images = []
    with ZipFile(path) as f:
        for z in range(160, 184):
            with Image.open(io.BytesIO(f.read(f"{st}/{st}_{z:04d}.tif"))) as im:
                assert im.size == SPEC[i][:2]
                images.append(np.asarray(im.crop((lo, 376, lo + 120, 632))))
    a = np.stack(images)
    if kind == "raw":
        assert a.dtype == np.uint16
        return a
    assert np.isin(a, [0, 255]).all()
    return a == 0


def get(cache, i, phase):
    rp, rs = verified(cache, i, "raw", False)
    sp, ss = verified(cache, i, "segmented", False)
    raw = part(rp, i, "raw", phase)
    mask = part(sp, i, "segmented", phase)
    return (
        raw,
        mask,
        dict(
            raw=rs["sha256"],
            segmented=ss["sha256"],
            raw_crop=digest(raw),
            label_crop=digest(mask),
        ),
    )


def sides(mask):
    interior = np.zeros_like(mask)
    interior[2:-2, 2:-2, 2:-2] = True
    return {
        "pore": interior & mask & ndi.binary_dilation(~mask, FACE),
        "material": interior & ~mask & ndi.binary_dilation(mask, FACE),
    }


def kernel(masses):
    q = np.asarray(masses, dtype="i8")
    assert len(q) == 5 and q.min() >= 0 and q.sum() == 4096
    w = q * (24 // COUNTS)
    grid = np.zeros(DIST.shape, "i8")
    for j, v in enumerate(w):
        grid[DIST == j] = v
    assert int(grid.sum()) == D
    return w, grid


def simplex(gram, rhs):
    best = None
    for bits in range(1, 32):
        ix = np.array([j for j in range(5) if bits & (1 << j)])
        n = len(ix)
        mat = np.ones((n + 1, n + 1))
        mat[:-1, :-1] = gram[np.ix_(ix, ix)]
        mat[-1, -1] = 0
        v = np.linalg.lstsq(mat, np.r_[rhs[ix], 1.0], rcond=None)[0][:-1]
        if v.min() < -1e-12 or abs(v.sum() - 1) > 1e-10:
            continue
        a = np.zeros(5)
        a[ix] = np.maximum(v, 0)
        a /= a.sum()
        key = (float(a @ gram @ a - 2 * rhs @ a), bits)
        if best is None or key < best[0]:
            best = (key, a)
    if best is None:
        raise ValueError("No simplex solution")
    a = best[1]
    f = 4096 * a
    q = np.floor(f).astype(int)
    q[np.lexsort((np.arange(5), q - f))[: 4096 - q.sum()]] += 1
    return a, q, best[0][1]


def fit(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    gram = np.zeros((5, 5))
    rhs = np.zeros(5)
    spec = {}
    valid = True
    for i in DEV:
        raw, t, r = get(cache, i, "fit")
        cal = calibration(raw, t)
        groups = sides(t)
        r.update(phases=cal, boundary={k: int(v.sum()) for k, v in groups.items()})
        spec[str(i)] = r
        if not cal["valid"] or min(r["boundary"].values()) == 0:
            valid = False
            continue
        m, p = (cal[k]["median_x2"] for k in ("material", "pore"))
        x = np.stack(
            [
                ndi.correlate(t.astype("u1"), (DIST == j).astype("u1"), mode="nearest")
                for j in range(5)
            ],
            axis=-1,
        )
        for sel in groups.values():
            y = (m - 2 * raw[sel].astype("f8")) / (m - p)
            v = x[sel] / COUNTS
            gram += v.T @ v / (6 * len(y))
            rhs += v.T @ y / (6 * len(y))
    a, q, bits = simplex(gram, rhs) if valid else (None, None, None)
    ident = identity()
    write(
        out / "fit.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            specimens=spec,
            valid=valid,
            active_set=bits,
            gram=gram.tolist(),
            rhs=rhs.tolist(),
            shell_masses=None if a is None else a.tolist(),
            quantized=None if q is None else q.tolist(),
        ),
    )
    print("valid", valid, "quantized", q, spec, flush=True)


def check(cache, out):
    fit = json.loads((E / "fit.json").read_bytes())
    ident = identity()
    assert ident == fit["identity"] and token(ident) == fit["pipeline_id"]
    records = {}
    checks = []
    if fit["valid"]:
        for i in DEV:
            raw, t, r = get(cache, i, "check")
            cal = fit["specimens"][str(i)]
            assert all(r[k] == cal[k] for k in ("raw", "segmented"))
            m, p = (cal["phases"][k]["median_x2"] for k in ("material", "pore"))
            y = (m - 2 * raw.astype("f8")) / (m - p)
            g = sides(t)
            arms = {}
            for k, q in [("center", CONTROL), ("blur", fit["quantized"])]:
                w, grid = kernel(q)
                v = native(t, w)
                cpu = ndi.correlate(t.astype("i4"), grid, mode="nearest")
                diff = int(np.count_nonzero(cpu != v))
                assert not diff
                err = (v.astype("f8") / D - y) ** 2
                mse = {s: float(err[sel].mean()) if sel.any() else None for s, sel in g.items()}
                arms[k] = dict(
                    rms={s: float(np.sqrt(x)) if x is not None else None for s, x in mse.items()},
                    balanced=float(np.sqrt(sum(mse.values()) / 2))
                    if None not in mse.values()
                    else None,
                    integer_field=digest(v),
                    cpu_gpu_differences=diff,
                )
            a, b = arms.values()
            counts = {k: int(sel.sum()) for k, sel in g.items()}
            c = dict(
                support=min(counts.values()) >= 128,
                balanced=a["balanced"] is not None and b["balanced"] <= 0.9 * a["balanced"],
                both_sides=all(a["rms"][s] is not None and b["rms"][s] <= a["rms"][s] for s in g),
            )
            checks.append(all(c.values()))
            records[str(i)] = dict(source=r, counts=counts, arms=arms, criteria=c)
            print(i, arms, c, flush=True)
    write(
        out,
        dict(
            fit_sha256=sha(E / "fit.json"),
            passed=fit["valid"] and len(checks) == 3 and all(checks),
            specimens=records,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Do not use -O")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("fit", "check"))
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    (fit if a.phase == "fit" else check)(a.cache, a.out)
