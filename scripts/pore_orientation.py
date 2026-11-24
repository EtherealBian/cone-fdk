"""Eight frozen XY views of the same census features; no label input or fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import (
    DEV,
    DILATIONS,
    PTX,
    ROOT,
    SHAPE,
    Denoise,
    Z,
    digest,
    features,
    forward,
    sha,
    token,
    verified,
    write,
)
from pore_axial_replay import cpu
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_census_train import identity as parent_identity

E = ROOT / "pores/evidence/orientation"
ARMS = ("identity", "mean8")


def orient(a, k, invert=False):
    if k not in range(8):
        raise ValueError("Expected one of eight orientations")
    if invert and k & 4:
        a = a.swapaxes(-1, -2)
    if k & 1:
        a = a[..., ::-1]
    if k & 2:
        a = a[..., ::-1, :]
    if not invert and k & 4:
        a = a.swapaxes(-1, -2)
    return a


def identity():
    files = (
        "pores/ORIENTATION-PILOT.md",
        "scripts/pore_orientation.py",
        "scripts/pore_orientation_score.py",
        "scripts/pore_axial_replay.py",
        "scripts/pore_section_score.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_transfer.py",
        "pores/validate.py",
        "pores/pore_check.py",
    )
    return dict(
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in files
        },
        parent=sha(EVID / "predictions.json"),
        parent_software=parent_identity(),
        weights=sha(EVID / "census.npz"),
        native={k: sha(PTX / k) for k in ("cone-pores.exe", "pores.ptx")},
    )


def predict(cache, out):
    ident = identity()
    parent = json.loads((EVID / "predictions.json").read_bytes())
    assert (
        parent["identity"] == ident["parent_software"]
        and parent["fits"]["census"]["weights"] == ident["weights"]
    )
    with np.load(EVID / "census.npz", allow_pickle=False) as f:
        pars = [(f[f"w{j}"], f[f"b{j}"], s) for j, s in enumerate(DILATIONS)]
    out.mkdir(parents=True, exist_ok=False)
    spec = {}
    for i in DEV:
        r = parent["specimens"][str(i)]
        src, rs = verified(cache, i, "raw", False)
        assert rs["sha256"] == r["raw"]
        base = read_mask(EVID, i, "census", r["masks"]["census"])
        v = np.zeros(SHAPE, "u1")
        mask = np.zeros(SHAPE, bool)
        hist = np.zeros(8, "i8")
        err = np.zeros(8)
        final = 0
        control = 0
        ferr = 0.0
        hh = hashlib.sha256()
        with Denoise(PTX / "search-support.ptx") as gpu:
            for off in range(0, SHAPE[0], 4):
                x = features(src, i, Z[0] + off, Z[0] + off + 4, gpu)
                hh.update(np.ascontiguousarray(x).view("u1"))
                prior = x[:, -1, 8:-8, 8:-8]
                ss = np.zeros_like(prior, dtype="f8")
                sc = np.zeros_like(ss)
                sub = np.s_[off : off + 4]
                for k in range(8):
                    a = np.ascontiguousarray(orient(x, k))
                    cuda = orient(forward(a, pars, PTX / "spatial.ptx")[:, 0], k, True) + prior
                    double = orient(np.stack([cpu(one, pars)[0] for one in a]), k, True) + prior
                    if not (np.isfinite(cuda).all() and np.isfinite(double).all()):
                        raise ValueError("Nonfinite logit")
                    pos = cuda >= 0
                    hist[k] += np.count_nonzero(pos != (double >= 0))
                    err[k] = max(err[k], float(np.max(abs(cuda - double))))
                    v[sub] += pos
                    ss += cuda
                    sc += double
                    if k == 0:
                        control += int(np.count_nonzero(pos != base[sub]))
                mask[sub] = ss >= 0
                final += int(np.sum(mask[sub] != (sc >= 0)))
                ferr = max(ferr, float(np.max(abs(ss - sc))) / 8)
                if off % 24 == 0:
                    print(i, off, "/96", flush=True)
        assert control == 0 and np.all(mask[v == 8]) and not np.any(mask[v == 0])
        rec = {}
        for k, m in zip(ARMS, (base, mask), strict=True):
            path = out / f"sample{i}-{k}.npz"
            with path.open("xb") as fp:
                np.savez_compressed(fp, mask=m)
            rec[k] = dict(
                file=sha(path),
                mask=digest(m),
                cpu_gpu_differences=int(hist[0]) if k == "identity" else final,
                additions=int(np.sum(m & ~base)),
                deletions=int(np.sum(~m & base)),
            )
        vp = out / f"sample{i}-votes.npz"
        with vp.open("xb") as fp:
            np.savez_compressed(fp, votes=v)
        spec[str(i)] = dict(
            raw=r["raw"],
            segmented=r["segmented"],
            features_f32=hh.hexdigest(),
            census=digest(base),
            masks=rec,
            votes_file=sha(vp),
            votes_u8=digest(v),
            vote_counts=np.bincount(v.ravel(), minlength=9).tolist(),
            view_cpu_gpu_differences=hist.tolist(),
            view_max_error=err.tolist(),
            mean_max_error=ferr,
            identity_parent_differences=control,
        )
        print(i, rec, "CPU/GPU", hist.tolist(), final, flush=True)
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            shape_zyx=list(SHAPE),
            origin_xyz=[360, 376, 256],
            specimens=spec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    predict(a.cache, a.out)
