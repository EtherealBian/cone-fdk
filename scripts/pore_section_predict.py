"""Generate and seal six area-screen masks; no transfer annotations are read here."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_section_area import DEV, train
from pore_shared_predict import model_id as h_identity
from pore_spatial_gpu import forward
from pore_spatial_model import ROOT, fields
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


def fit(cache, out):

    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use empty output directory")
    ident = identity()
    x, t, scans = {}, {}, {}
    with Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as gpu:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
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
        _, trained = train(x, t, out)
    doc = dict(
        identity=ident,
        pipeline_id=token(ident),
        specimens=scans,
        training=trained,
        weights={k: sha(out / f"{k}.npz") for k in ("control", "area")},
    )
    (out / "fit.json").write_text(json.dumps(doc, allow_nan=False) + "\n", encoding="utf8")


def infer(cache, out, i):
    import torch
    from pore_field_replay import load_layers
    from pore_spatial_model import network

    if i not in DEV:
        raise ValueError("Development only")
    f = json.loads((out / "fit.json").read_text())
    ident = identity()
    if f["identity"] != ident or f["pipeline_id"] != token(ident):
        raise ValueError("Fit protocol changed")
    arms = ("control", "area")
    pars = {}
    nets = {}
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    for k in arms:
        wp = out / f"{k}.npz"
        if sha(wp) != f["weights"][k]:
            raise ValueError("Different trained weights")
        pars[k] = load_layers(wp, "H")
        net = network()
        nets[k] = net.eval()
        layers = [v for v in net if hasattr(v, "weight")]
        for layer, (w, b, _) in zip(layers, pars[k], strict=True):
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(w))
                layer.bias.copy_(torch.from_numpy(b))
    raw, rs = verified(cache, i, "raw", False)
    if rs["sha256"] != f["specimens"][str(i)]["raw"]:
        raise ValueError("Different raw source")
    record = {}
    batch = 2
    with Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as gpu:
        with torch.inference_mode():
            masks = {k: np.empty(SHAPE, bool) for k in arms}
            flips = {k: 0 for k in arms}
            err = {k: 0.0 for k in arms}
            for a in range(*Z, batch):
                v = fields(raw, i, a, a + batch, gpu)
                prior = v[:, -1, 8:-8, 8:-8]
                tv = torch.from_numpy(v)
                for k in arms:
                    n = forward(v, pars[k], ROOT / "build/pores/spatial.ptx")[:, 0] + prior
                    c = nets[k](tv).numpy()[:, 0] + prior
                    if (
                        n.shape != (batch, 256, 256)
                        or not np.isfinite(n).all()
                        or not np.isfinite(c).all()
                    ):
                        raise ValueError("Invalid prediction field")
                    masks[k][a - Z[0] : a - Z[0] + batch] = n >= 0
                    flips[k] += int(((c >= 0) != (n >= 0)).sum())
                    err[k] = max(err[k], float(np.max(np.abs(c - n))))
            for k in arms:
                path = out / f"sample{i}-{k}.npy"
                np.save(path, masks[k], allow_pickle=False)
                record[k] = dict(
                    file=sha(path),
                    mask=digest(masks[k]),
                    decision_disagreements=flips[k],
                    max_abs_cpu_gpu_logit=err[k],
                )
            print("sealed", i, record, flush=True)
    (out / f"scan{i}.json").write_text(
        json.dumps(dict(fit=sha(out / "fit.json"), masks=record), allow_nan=False) + "\n",
        encoding="utf8",
    )


def seal(out):
    fit = json.loads((out / "fit.json").read_text())
    ident = identity()
    if fit["identity"] != ident or fit["pipeline_id"] != token(ident):
        raise ValueError("Different protocol")
    for i in DEV:
        s = json.loads((out / f"scan{i}.json").read_text())
        if s["fit"] != sha(out / "fit.json") or set(s["masks"]) != {"area", "control"}:
            raise ValueError("Partial or stale scan predictions")
        for k, rec in s["masks"].items():
            p = out / f"sample{i}-{k}.npy"
            m = np.load(p, mmap_mode="r")
            if (
                sha(p) != rec["file"]
                or digest(m) != rec["mask"]
                or m.shape != SHAPE
                or m.dtype != bool
            ):
                raise ValueError("Malformed scan output")
        fit["specimens"][str(i)]["masks"] = s["masks"]
    fit.update(origin_xyz=[360, 376, Z[0]], shape_zyx=list(SHAPE))
    with (out / "predictions.json").open("x", encoding="utf8") as fp:
        json.dump(fit, fp, indent=2, allow_nan=False)
        fp.write("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["fit", "infer", "seal"])
    ap.add_argument("--specimen", type=int, choices=DEV)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    if a.phase == "fit":
        fit(a.cache, a.out)
    elif a.phase == "infer":
        infer(a.cache, a.out, a.specimen)
    else:
        seal(a.out)
