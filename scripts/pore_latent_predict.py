"""Fit candidate G on the original calibration only and seal all development decision margins."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pore_latent_model import cpu_decision, cuda_decision, train
from pore_spatial_gpu import forward
from pore_spatial_model import ROOT, fields, parameters
from pore_transfer import DEV, digest

sys.path.insert(0, str(ROOT / "pores"))
from denoise import Denoise
from freeze import identity, token
from local_model import SETTINGS
from validate import sha, stack, verified

Z = (400, 528)
SHAPE = (128, 256, 256)
NAME = "latent intensity Bernsen G"
SEAL = ROOT / "pores/evidence/candidate-g/predictions.json"


def model_id():
    import torch

    files = {
        f: sha(ROOT / f)
        if f.endswith(".ptx")
        else hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for f in [
            "pores/EXPERIMENT-F.md",
            "pores/spatial.cu",
            "build/pores/spatial.ptx",
            "scripts/pore_spatial_gpu.py",
            "scripts/pore_spatial_model.py",
            "scripts/pore_latent_predict.py",
            "scripts/pore_latent_model.py",
            "pores/EXPERIMENT-G.md",
            "pores/latent.cu",
            "build/pores/latent.ptx",
        ]
    }
    return dict(base=identity("E"), files=files, torch=torch.__version__)


def generate(cache, out):
    import torch

    if not __debug__:
        raise ValueError("Do not run with -O")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use empty output directory")
    ident = model_id()
    records = {}
    torch.set_num_threads(3)
    with (
        Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as gpu,
        Denoise(ROOT / "build/pores/latent.ptx") as decision_gpu,
    ):
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            print("G", i, "calibrating on original 24 slices", flush=True)
            x = np.concatenate([fields(raw, i, a, a + 4, gpu) for a in range(160, 184, 4)])
            t = stack(seg, i, "segmented", 160, 184)
            ct = SETTINGS[i][2]
            net, losses = train(x, t, ct)
            del x, t
            pars = parameters(net)
            weights = out / f"sample{i}-weights.npz"
            np.savez(
                weights,
                **{
                    f"{k}{n}": v
                    for n, (w, b, _) in enumerate(pars)
                    for k, v in [("w", w), ("b", b)]
                },
            )
            dest = out / f"sample{i}.f32"
            mismatches = 0
            maxdiff = 0
            first = None
            quant_diff = 0
            with torch.inference_mode(), dest.open("xb") as fp:
                for a in range(*Z, 4):
                    x = fields(raw, i, a, a + 4, gpu)
                    base = x[:, 3, 3:-3, 3:-3] * 48 + 128
                    corr = forward(x, pars, ROOT / "build/pores/spatial.ptx")[:, 0]
                    ng, nq = cuda_decision(decision_gpu, base, corr, ct)
                    cg, cq = cpu_decision(base, net(torch.from_numpy(x)).numpy()[:, 0], ct)
                    quant_diff += int(np.count_nonzero(nq != cq))
                    native, cpu = [m[:, 5:-5, 5:-5] for m in (ng, cg)]
                    if (
                        native.shape != (4, *SHAPE[1:])
                        or not np.isfinite(native).all()
                        or not np.isfinite(cpu).all()
                    ):
                        raise ValueError("Invalid logits")
                    maxdiff = max(maxdiff, float(np.abs(native - cpu).max()))
                    neq = (native >= 0) != (cpu >= 0)
                    mismatches += int(neq.sum())
                    if neq.any() and first is None:
                        z, y, u = np.argwhere(neq)[0]
                        first = [
                            int(u) + 360,
                            int(y) + 376,
                            int(z) + a,
                            float(cpu[z, y, u]),
                            float(native[z, y, u]),
                        ]
                    native.astype("<f4").tofile(fp)
            logits = np.memmap(dest, mode="r", dtype="<f4", shape=SHAPE)
            records[str(i)] = dict(
                raw=rs["sha256"],
                segmented=ss["sha256"],
                weights=sha(weights),
                field=sha(dest),
                mask=digest(logits >= 0),
                max_abs_cpu_gpu_margin=maxdiff,
                quantized_intensity_disagreements_including_halo=quant_diff,
                cpu_gpu_decision_disagreements=mismatches,
                first_disagreement=first,
                calibration_training_trace=losses,
            )
            print("G sealed", i, records[str(i)], flush=True)
    (out / "predictions.json").write_text(
        json.dumps(
            dict(
                schema=1,
                pipeline_id=token(ident),
                identity=ident,
                origin_xyz=[360, 376, 400],
                shape_zyx=list(SHAPE),
                specimens=records,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    generate(a.cache, a.out)
