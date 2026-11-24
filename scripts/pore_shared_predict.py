"""Train one development-only H model; no formal labels are decoded in this program."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from pore_shared_model import DEV, train
from pore_spatial_gpu import forward
from pore_spatial_model import ROOT, fields, parameters
from pore_spatial_predict import Denoise, SHAPE, Z, digest, sha, stack, token, verified
from pore_spatial_predict import model_id as f_id

SEAL = ROOT / "pores/evidence/candidate-h/predictions.json"
NAME = "pooled complete-pore field H"


def model_id():
    files = [
        "pores/EXPERIMENT-H.md",
        "scripts/pore_shared_model.py",
        "scripts/pore_shared_predict.py",
    ]
    return dict(
        parent=f_id(),
        files={
            f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for f in files
        },
    )


def generate(cache, out):
    import torch

    if not __debug__:
        raise ValueError("Do not run with -O")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new directory")
    ident = model_id()
    x, t, raws, records = {}, {}, {}, {}
    with Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as gpu:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            raws[i] = raw
            records[str(i)] = dict(raw=rs["sha256"], segmented=ss["sha256"])
            x[i] = np.concatenate([fields(raw, i, a, a + 4, gpu) for a in range(160, 184, 4)])
            t[i] = stack(seg, i, "segmented", 160, 184)
            print("calibration ready", i, flush=True)
        net, fit = train(x, t)
        del x, t
        pars = parameters(net)
        wp = out / "shared-weights.npz"
        np.savez(
            wp,
            **{f"{k}{j}": v for j, (w, b, _) in enumerate(pars) for k, v in [("w", w), ("b", b)]},
        )
        for i in DEV:
            shutil.copyfile(wp, out / f"sample{i}-weights.npz")
            path = out / f"sample{i}.f32"
            flips = 0
            err = 0.0
            first = None
            with path.open("xb") as fp, torch.inference_mode():
                for a in range(*Z, 4):
                    x = fields(raws[i], i, a, a + 4, gpu)
                    prior = x[:, -1, 8:-8, 8:-8]
                    native = forward(x, pars, ROOT / "build/pores/spatial.ptx")[:, 0] + prior
                    cpu = net(torch.from_numpy(x)).numpy()[:, 0] + prior
                    if (
                        native.shape != (4, *SHAPE[1:])
                        or not np.isfinite(native).all()
                        or not np.isfinite(cpu).all()
                    ):
                        raise ValueError("Invalid field")
                    mismatch = (native >= 0) != (cpu >= 0)
                    flips += int(mismatch.sum())
                    err = max(err, float(np.abs(native - cpu).max()))
                    if mismatch.any() and first is None:
                        z, y, u = np.argwhere(mismatch)[0]
                        first = [
                            int(u) + 360,
                            int(y) + 376,
                            int(z) + a,
                            float(cpu[z, y, u]),
                            float(native[z, y, u]),
                        ]
                    native.astype("<f4").tofile(fp)
            field = np.memmap(path, mode="r", dtype="<f4", shape=SHAPE)
            records[str(i)].update(
                weights=sha(wp),
                logits=sha(path),
                mask=digest(field >= 0),
                max_abs_cpu_gpu_logit=err,
                cpu_gpu_decision_disagreements=flips,
                first_disagreement=first,
            )
            print("H sealed", i, records[str(i)], flush=True)
    doc = dict(
        schema=1,
        pipeline_id=token(ident),
        identity=ident,
        origin_xyz=[360, 376, 400],
        shape_zyx=list(SHAPE),
        shared_calibration=fit,
        specimens=records,
    )
    (out / "predictions.json").write_text(
        json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    generate(a.cache, a.out)
