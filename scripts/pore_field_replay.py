"""Replay stored scan weights from authenticated raw intensities, separately from scoring.

Only development 2/3/5 are accepted. Source features share the frozen CUDA NLM,
while the convolution and hard-decision replay use CPU Torch/SciPy. Compare every
voxel with both the published field and the offline viewer. Never refit weights.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from pore_latent_model import cpu_decision
from pore_spatial_model import DILATIONS, ROOT, fields
from pore_spatial_predict import DEV, SHAPE, Denoise, Z, digest, sha, token, verified


def load_layers(path, model):
    if model not in ("F", "G", "H"):
        raise ValueError("Unknown model")
    dims = [22, 16, 16, 16, 16, 1] if model in ("F", "H") else [22, 12, 12, 12, 1]
    layers = []
    with np.load(path, allow_pickle=False) as f:
        n = len(dims) - 1
        if set(f.files) != {k + str(j) for j in range(n) for k in ("b", "w")}:
            raise ValueError("Wrong weight keys")
        for j, (ci, co) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            w, b = f[f"w{j}"], f[f"b{j}"]
            k = 1 if j == n - 1 else 3
            if (
                w.dtype != np.float32
                or b.dtype != np.float32
                or w.shape != (co, ci, k, k)
                or b.shape != (co,)
                or not np.isfinite(w).all()
                or not np.isfinite(b).all()
            ):
                raise ValueError("Malformed scan weights")
            layers.append((w, b, DILATIONS[j] if model in ("F", "H") else 1))
    return layers


def masks(view):
    from audit import decode

    match = re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        view.read_text(encoding="utf8"),
        re.S,
    )
    if not match:
        raise ValueError("Missing viewer")
    p = json.loads(match[1])
    return p["layers"], p["shape"], p["origin"], p["specimen"], decode


def replay(model, folder, formal, cache):
    import torch
    from local_model import SETTINGS
    from torch.nn import functional as fn

    if model == "G":
        from pore_latent_predict import SEAL, model_id
    elif model == "H":
        from pore_shared_predict import SEAL, model_id
    elif model == "F":
        from pore_spatial_predict import SEAL, model_id
    else:
        raise ValueError("Unknown model")
    torch.manual_seed(61429)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    sealed = json.loads(SEAL.read_text())
    identity = model_id()
    if (
        sealed != json.loads((folder / "predictions.json").read_text())
        or sealed["identity"] != identity
        or sealed["pipeline_id"] != token(identity)
        or set(sealed["specimens"]) != {str(k) for k in DEV}
    ):
        raise ValueError("Source, environment or seal differs")
    if model == "H":
        from pore_shared_score import shared_digest

        shared_digest(sealed)
    result = {}
    with torch.inference_mode(), Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as gpu:
        for i in DEV:
            rec = sealed["specimens"][str(i)]
            s = f"sample{i}"
            wp = folder / (s + "-weights.npz")
            field = folder / (s + ".f32")
            native_key = "logits" if model in ("F", "H") else "field"
            if (
                sha(wp) != rec["weights"]
                or sha(field) != rec[native_key]
                or field.stat().st_size != np.prod(SHAPE) * 4
            ):
                raise ValueError("Unsealed weights or field")
            native = np.memmap(field, mode="r", dtype="<f4", shape=SHAPE)
            if not np.isfinite(native).all() or digest(native >= 0) != rec["mask"]:
                raise ValueError("Invalid stored mask")
            view = formal / (s + "-review.html")
            report = json.loads((formal / (s + ".json")).read_text())
            layers, shape, origin, num, decode = masks(view)
            if (
                num != i
                or shape != list(SHAPE)
                or len(layers) != SHAPE[0]
                or origin != sealed["origin_xyz"]
                or report["seal"] != rec
                or report["pipeline_id"] != sealed["pipeline_id"]
                or sha(view) != report["review_sha256"]
            ):
                raise ValueError("Unexpected review binding")
            pars = [
                (torch.from_numpy(w), torch.from_numpy(b), d)
                for w, b, d in load_layers(wp, model)
            ]
            raw, rs = verified(cache, i, "raw", False)
            if rs["sha256"] != rec["raw"]:
                raise ValueError("Different raw archive")
            flips = 0
            maxdiff = 0.0
            for a in range(*Z, 4):
                x = fields(raw, i, a, a + 4, gpu)
                v = torch.from_numpy(x)
                for j, (w, b, d) in enumerate(pars):
                    v = fn.conv2d(v, w, b, dilation=d)
                    if j < len(pars) - 1:
                        v = v.relu()
                v = v.numpy()[:, 0]
                if model in ("F", "H"):
                    pred = v + x[:, -1, 8:-8, 8:-8]
                else:
                    pred = cpu_decision(x[:, 3, 3:-3, 3:-3] * 48 + 128, v, SETTINGS[i][2])[0][
                        :, 5:-5, 5:-5
                    ]
                z = a - Z[0]
                src = native[z : z + len(x)]
                maxdiff = max(maxdiff, float(np.abs(src - pred).max()))
                flips += int(np.count_nonzero((src >= 0) != (pred >= 0)))
                for j, plane in enumerate(src):
                    ids = decode(layers[z + j]["objects"], 65536).reshape(256, 256)
                    if not np.array_equal(plane >= 0, ids != 0):
                        raise ValueError("Viewer mask differs from sealed field")
            expected = rec[
                "max_abs_cpu_gpu_logit" if model in ("F", "H") else "max_abs_cpu_gpu_margin"
            ]
            if flips != rec["cpu_gpu_decision_disagreements"] or maxdiff != expected:
                raise ValueError(
                    f"CPU replay differs from prediction transcript: {i}, {flips}, {maxdiff}, {expected}"
                )
            result[str(i)] = dict(
                voxels=int(np.prod(SHAPE)),
                mask=rec["mask"],
                decision_disagreements=flips,
                max_field_difference=maxdiff,
                raw=rec["raw"],
            )
            print(model, i, result[str(i)], flush=True)
    return dict(pipeline_id=sealed["pipeline_id"], model=model, specimens=result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["F", "G", "H"], required=True)
    for k in ("predictions", "formal", "cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    r = replay(a.model, a.predictions, a.formal, a.cache)
    with a.out.open("x", encoding="utf8") as f:
        json.dump(r, f, indent=2)
        f.write("\n")
