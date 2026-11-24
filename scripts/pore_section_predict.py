"""Generate and seal six area-screen masks; no transfer annotations are read here."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_section_area import DEV, train
from pore_shared_predict import model_id as h_identity
from pore_spatial_gpu import forward
from pore_spatial_model import ROOT, fields, parameters
from pore_spatial_predict import Denoise, digest, sha, stack, token, verified

Z = (256, 352)
SHAPE = (96, 256, 256)


def identity():
    import hashlib

    return {
        "parent": h_identity(),
        "files": {
            f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for f in (
                "pores/SECTION-AREA-PILOT.md",
                "scripts/pore_section_area.py",
                "scripts/pore_section_predict.py",
            )
        },
    }


def generate(cache, out):
    import torch

    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use empty output directory")
    ident = identity()
    x, t, raws, scans = {}, {}, {}, {}
    with Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as gpu:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            raws[i] = raw
            path = out / f"cal-{i}.npy"
            arr = np.lib.format.open_memmap(
                path, mode="w+", dtype="<f4", shape=(24, 22, 272, 272)
            )
            for a in range(160, 184, 4):
                arr[a - 160 : a - 156] = fields(raw, i, a, a + 4, gpu)
            arr.flush()
            del arr
            x[i] = np.load(path, mmap_mode="r")
            t[i] = stack(seg, i, "segmented", 160, 184)
            scans[i] = dict(
                raw=rs["sha256"],
                segmented=ss["sha256"],
                features=sha(path),
                calibration_mask=digest(t[i]),
                masks={},
            )
        nets, fit = train(x, t, out)
        del x, t
        arms = ("control", "area")
        pars = {name: parameters(nets[name]) for name in arms}
        with torch.inference_mode():
            for i in DEV:
                masks = {k: np.empty(SHAPE, bool) for k in arms}
                flips = {k: 0 for k in arms}
                err = {k: 0.0 for k in arms}
                for a in range(*Z, 4):
                    v = fields(raws[i], i, a, a + 4, gpu)
                    prior = v[:, -1, 8:-8, 8:-8]
                    tv = torch.from_numpy(v)
                    for k in arms:
                        n = forward(v, pars[k], ROOT / "build/pores/spatial.ptx")[:, 0] + prior
                        c = nets[k](tv).numpy()[:, 0] + prior
                        if (
                            n.shape != (4, 256, 256)
                            or not np.isfinite(n).all()
                            or not np.isfinite(c).all()
                        ):
                            raise ValueError("Invalid prediction field")
                        masks[k][a - Z[0] : a - Z[0] + 4] = n >= 0
                        flips[k] += int(((c >= 0) != (n >= 0)).sum())
                        err[k] = max(err[k], float(np.max(np.abs(c - n))))
                for k in arms:
                    path = out / f"sample{i}-{k}.npy"
                    np.save(path, masks[k], allow_pickle=False)
                    scans[i]["masks"][k] = dict(
                        file=sha(path),
                        mask=digest(masks[k]),
                        decision_disagreements=flips[k],
                        max_abs_cpu_gpu_logit=err[k],
                    )
                print("sealed", i, scans[i]["masks"], flush=True)
    result = dict(
        identity=ident,
        pipeline_id=token(ident),
        origin_xyz=[360, 376, Z[0]],
        shape_zyx=list(SHAPE),
        weights={k: sha(out / f"{k}.npz") for k in arms},
        training=fit,
        specimens=scans,
    )
    (out / "predictions.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    generate(a.cache, a.out)
