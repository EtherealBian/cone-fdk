"""Low-commit replay: single raw plane, native CUDA and an independent NumPy CPU convolution."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pore_axial_raw import (
    ARMS,
    DEV,
    DILATIONS,
    PTX,
    ROOT,
    SHAPE,
    Denoise,
    Z,
    arm_input,
    digest,
    features,
    forward,
    sha,
    token,
    verified,
    write,
)

FILES = (
    "pores/AXIAL-RAW-PILOT.md",
    "scripts/pore_axial_raw.py",
    "scripts/pore_shared_model.py",
    "scripts/pore_spatial_model.py",
    "scripts/pore_spatial_gpu.py",
    "pores/search_support.cu",
    "pores/spatial.cu",
    "pores/local_model.py",
    "pores/denoise.py",
    "pores/validate.py",
)


def check_identity(d):
    ident = dict(
        files={
            f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for f in FILES
        },
        ptx={f: sha(PTX / f) for f in ("search-support.ptx", "spatial.ptx")},
        torch=version("torch"),
    )
    if ident != d["identity"] or token(ident) != d["pipeline_id"]:
        raise ValueError("Changed registered fit identity")


def cpu(x, layers):
    # Accumulate in double: order and precision are independent of CUDA and torch.
    x = np.asarray(x, dtype="f8")
    for j, (w, b, d) in enumerate(layers):
        co, ci, k, _ = w.shape
        ny = x.shape[1] - (k - 1) * d
        nx = x.shape[2] - (k - 1) * d
        out = np.broadcast_to(b[:, None, None], (co, ny, nx)).astype("f8").reshape(co, -1)
        weights = w.astype("f8")
        v = np.empty_like(out)
        for y in range(k):
            for u in range(k):
                left = weights[:, :, y, u].copy()
                right = np.ascontiguousarray(
                    x[:, y * d : y * d + ny, u * d : u * d + nx]
                ).reshape(ci, -1)
                np.matmul(left, right, out=v)
                out += v
        if j < len(layers) - 1:
            np.maximum(out, 0, out=out)
        x = out.reshape(co, ny, nx)
    return x


def infer(cache, out, i):
    if i not in DEV:
        raise ValueError("Development only")
    doc = json.loads((out / "fit.json").read_text())
    check_identity(doc)
    pars = {}
    for k in ARMS:
        wp = out / f"{k}.npz"
        if sha(wp) != doc["fits"][k]["weights"]:
            raise ValueError("Changed weights")
        with np.load(wp, allow_pickle=False) as f:
            pars[k] = [(f[f"w{j}"], f[f"b{j}"], d) for j, d in enumerate(DILATIONS)]
    raw, rs = verified(cache, i, "raw", False)
    rec = doc["specimens"][str(i)]
    if rs["sha256"] != rec["raw"]:
        raise ValueError("Changed source")
    flips = dict.fromkeys(ARMS, 0)
    errors = dict.fromkeys(ARMS, 0.0)
    first = dict.fromkeys(ARMS, None)
    masks = {k: np.empty(SHAPE, bool) for k in ARMS}
    base = hashlib.sha256()
    with Denoise(PTX / "search-support.ptx") as gpu:
        for z in range(*Z):
            x = features(raw, i, z, z + 1, gpu)
            prior = x[0, -1, 8:-8, 8:-8]
            base.update((prior >= 0).tobytes())
            for k in ARMS:
                v = arm_input(x, k)
                n = forward(v, pars[k], PTX / "spatial.ptx")[0, 0] + prior
                c = cpu(v[0], pars[k])[0] + prior
                if not np.isfinite(n).all() or not np.isfinite(c).all():
                    raise ValueError("Non-finite logits")
                change = (n >= 0) != (c >= 0)
                masks[k][z - Z[0]] = n >= 0
                flips[k] += int(change.sum())
                errors[k] = max(errors[k], float(abs(c - n).max()))
                if change.any() and first[k] is None:
                    y, u = np.argwhere(change)[0]
                    first[k] = [int(u) + 360, int(y) + 376, z, float(n[y, u]), float(c[y, u])]
            if (z - Z[0]) % 24 == 0:
                print(i, z, flush=True)
    rec["fixed_baseline"] = base.hexdigest()
    rec["masks"] = {}
    for k in ARMS:
        path = out / f"sample{i}-{k}.npz"
        if path.exists():
            with np.load(path) as f:
                if not np.array_equal(f["mask"], masks[k]):
                    raise ValueError("Previous native inference differs!")
        else:
            np.savez_compressed(path, mask=masks[k])
        rec["masks"][k] = dict(
            file=sha(path),
            mask=digest(masks[k]),
            cpu_gpu_decision_disagreements=flips[k],
            max_abs_cpu_gpu=errors[k],
            first_difference_xyz_native_cpu=first[k],
        )
        print(i, k, rec["masks"][k], flush=True)
    write(out / f"replay{i}.json", dict(fit=sha(out / "fit.json"), record=rec))


def seal(out):
    d = json.loads((out / "fit.json").read_text())
    check_identity(d)
    for i in DEV:
        v = json.loads((out / f"replay{i}.json").read_text())
        r = v["record"]
        if v["fit"] != sha(out / "fit.json") or any(
            r[k] != val for k, val in d["specimens"][str(i)].items()
        ):
            raise ValueError("Unrelated replay")
        for k in ARMS:
            p = out / f"sample{i}-{k}.npz"
            with np.load(p) as f:
                m = f["mask"]
            if (
                sha(p) != r["masks"][k]["file"]
                or m.shape != SHAPE
                or digest(m) != r["masks"][k]["mask"]
                or m.dtype != bool
            ):
                raise ValueError("Different replay mask")
        d["specimens"][str(i)] = r
    d.update(
        origin_xyz=[360, 376, Z[0]],
        shape_zyx=list(SHAPE),
        replay="native one-plane vs independent FP64 CPU",
        replay_code=hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
    )
    write(out / "predictions.json", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--specimen", type=int, choices=DEV)
    a = ap.parse_args()
    infer(a.cache, a.out, a.specimen) if a.specimen else seal(a.out)
