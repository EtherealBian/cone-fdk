"""Raw-only transfer replay of the publicly registered signed connectivity experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import (
    DEV,
    DILATIONS,
    PTX,
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
from pore_local_train import ARMS, OLD, identity


def run(cache, out):
    d = json.loads((out / "fit.json").read_text())
    ident = identity()
    if (
        ident != d["identity"]
        or token(ident) != d["pipeline_id"]
        or set(d["specimens"]) != set("235")
    ):
        raise ValueError("Changed fit")
    control = json.loads((OLD / "predictions.json").read_text())
    pars = {}
    for k in ARMS:
        p = out / f"{k}.npz"
        if sha(p) != d["fits"][k]["weights"]:
            raise ValueError("Changed weights")
        with np.load(p, allow_pickle=False) as f:
            pars[k] = [(f[f"w{j}"], f[f"b{j}"], s) for j, s in enumerate(DILATIONS)]
    for i in DEV:
        r = d["specimens"][str(i)]
        raw, rs = verified(cache, i, "raw", False)
        if rs["sha256"] != r["raw"]:
            raise ValueError("Different raw archive")
        masks = {k: np.empty(SHAPE, bool) for k in ARMS}
        flips = dict.fromkeys(ARMS, 0)
        errors = dict.fromkeys(ARMS, 0.0)
        first = dict.fromkeys(ARMS, None)
        with Denoise(PTX / "search-support.ptx") as gpu:
            for z in range(*Z):
                x = features(raw, i, z, z + 1, gpu)
                base = x[0, -1, 8:-8, 8:-8]
                for k in ARMS:
                    n = forward(x, pars[k], PTX / "spatial.ptx")[0, 0] + base
                    c = cpu(x[0], pars[k])[0] + base
                    if not (np.isfinite(n).all() and np.isfinite(c).all()):
                        raise ValueError("Nonfinite prediction")
                    m = n >= 0
                    changed = m != (c >= 0)
                    masks[k][z - Z[0]] = m
                    flips[k] += int(changed.sum())
                    errors[k] = max(errors[k], float(abs(n - c).max()))
                    if first[k] is None and changed.any():
                        y, u = np.argwhere(changed)[0]
                        first[k] = [int(u) + 360, int(y) + 376, z, float(n[y, u]), float(c[y, u])]
                if (z - Z[0]) % 24 == 0:
                    print(i, z, flush=True)
        r["masks"] = {}
        for k in ARMS:
            if (
                k == "census"
                and digest(masks[k]) != control["specimens"][str(i)]["masks"][k]["mask"]
            ):
                raise ValueError("Raw replay differs from the published census control")
            f = out / f"sample{i}-{k}.npz"
            with f.open("xb") as fp:
                np.savez_compressed(fp, mask=masks[k])
            r["masks"][k] = dict(
                file=sha(f),
                mask=digest(masks[k]),
                max_abs_cpu_gpu=errors[k],
                cpu_gpu_decision_disagreements=flips[k],
                first_difference=first[k],
            )
            print(i, k, r["masks"][k], flush=True)
    d.update(origin_xyz=[360, 376, Z[0]], shape_zyx=list(SHAPE))
    write(out / "predictions.json", d)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
