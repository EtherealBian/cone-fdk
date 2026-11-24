"""Matched plane/axial residual inference. No evaluation labels in fit or predict."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pore_patch_stats import DEV, ROOT, SETTINGS, Denoise, stack, verified
from pore_shared_model import region_logits, tiles
from pore_spatial_gpu import forward
from pore_spatial_model import DILATIONS, SEED, parameters, pools
from pore_spatial_predict import digest, sha, token
from scipy import ndimage as ndi

ARMS = ("plane", "axial")
Z = (256, 352)
SHAPE = (96, 256, 256)
SEAL = ROOT / "pores/evidence/axial-raw/predictions.json"
PTX = ROOT / "build/pores"


def identity():
    import torch

    names = (
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
    return dict(
        files={
            f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for f in names
        },
        ptx={f: sha(PTX / f) for f in ("search-support.ptx", "spatial.ptx")},
        torch=torch.__version__,
    )


def raw_pairs(volume, n):
    if volume.shape[0] != n + 6 or volume.dtype != np.float32:
        raise ValueError("Expected real three-plane context and mapped float32 data")
    c = volume[3 : 3 + n]
    result = [(c - 128) / 48]
    for k in (1, 3):
        lo, hi = volume[3 - k : 3 - k + n], volume[3 + k : 3 + k + n]
        result.extend((((lo + hi) * 0.5 - c) / 24, np.abs(lo - hi) / 48))
    return result


def features(path, i, a, b, gpu):
    from local_model import context_stack

    if i not in DEV or not (160 <= a < b <= 184 or Z[0] <= a < b <= Z[1]):
        raise ValueError("Not a registered development region")
    scale, shift, ct = SETTINGS[i]
    n = b - a
    raw = context_stack(path, i, a - 2, b + 2, 26)
    med = ndi.median_filter(raw[2:-2], size=3, mode="nearest")[1:-1]
    img = (gpu(med, 30 / scale) + shift) * scale
    u = np.clip(np.floor(img + 0.5), 0, 255).astype("i2")
    y, x = np.ogrid[-5:6, -5:6]
    disk = (x * x + y * y <= 26)[None]
    lo = ndi.minimum_filter(u, footprint=disk)
    hi = ndi.maximum_filter(u, footprint=disk)
    mid = (hi + lo) // 2
    span = hi - lo
    prior = np.clip(np.where(span < ct, 127.5 - mid, mid - u - 0.5), -4, 4)
    sl = np.s_[:, 18:-18, 18:-18]
    ch = raw_pairs((raw[sl].astype("f4") + shift) * scale, n)
    ch += [
        ((med[sl].astype("f4") + shift) * scale - 128) / 48,
        (img[sl] - 128) / 48,
        (span[sl] - ct) / 6,
        (mid[sl] - u[sl]) / 4,
        prior[sl],
    ]
    v = np.stack(ch, axis=1).astype("f4")
    assert v.shape == (n, 10, 272, 272) and np.isfinite(v).all()
    return v


def network():
    from torch import nn

    layers = []
    for j, d in enumerate(DILATIONS):
        last = j == 4
        layer = nn.Conv2d(10 if j == 0 else 16, 1 if last else 16, 1 if last else 3, dilation=d)
        layers.append(layer)
        if last:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        else:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def configure():
    import torch

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def arm_input(x, k):
    if k not in ARMS or x.shape[1] != 10:
        raise ValueError("Unknown arm or channels")
    if k == "axial":
        return x
    y = x.copy()
    y[:, 1:5] = 0
    return y


def fit(cache, out):
    import torch
    from torch.nn import functional as fn

    out.mkdir(parents=True, exist_ok=False)
    configure()
    ident = identity()
    x = {}
    t = {}
    rec = {}
    with Denoise(PTX / "search-support.ptx") as gpu:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            path = out / f"cal-{i}.npy"
            v = np.lib.format.open_memmap(path, mode="w+", dtype="<f4", shape=(24, 10, 272, 272))
            for a in range(160, 184, 4):
                v[a - 160 : a - 156] = features(raw, i, a, a + 4, gpu)
            v.flush()
            del v
            x[i] = np.load(path, mmap_mode="r")
            t[i] = stack(seg, i, "segmented", 160, 184)
            rec[i] = dict(
                raw=rs["sha256"],
                segmented=ss["sha256"],
                cal_features=sha(path),
                cal_mask=digest(t[i]),
            )
    sets = {i: pools(t[i]) for i in DEV}
    fits = {}
    for k in ARMS:
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        net = network()
        init = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in net.parameters())
        ).hexdigest()
        h = hashlib.sha256()
        trace = []
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1799, eta_min=0.0001)
        for step in range(1800):
            xx, tt = tiles(x, t, sets, step, rng)
            h.update(xx.tobytes())
            h.update(tt.tobytes())
            tt = torch.from_numpy(tt)
            v = region_logits(net, torch.from_numpy(arm_input(xx, k)))
            loss = (
                fn.binary_cross_entropy_with_logits(v, tt)
                + 0.1 * ((v.sigmoid() - tt).sum((2, 3)).abs() / (16 + tt.sum((2, 3)))).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 300 == 0 or step == 1799:
                trace.append([step, float(loss.detach())])
                print(k, trace[-1], flush=True)
        path = out / f"{k}.npz"
        np.savez(
            path,
            **{
                f"{s}{j}": z
                for j, (w, b, _) in enumerate(parameters(net))
                for s, z in [("w", w), ("b", b)]
            },
        )
        fits[k] = dict(init=init, stream=h.hexdigest(), weights=sha(path), trace=trace)
    if any(fits["plane"][k] != fits["axial"][k] for k in ("init", "stream")):
        raise ValueError("Not matched training")
    write(
        out / "fit.json", dict(identity=ident, pipeline_id=token(ident), specimens=rec, fits=fits)
    )


def write(p, d):
    with p.open("x", encoding="utf8", newline="\n") as fp:
        json.dump(d, fp, indent=2, allow_nan=False)
        fp.write("\n")


def infer(cache, out):
    import torch

    configure()
    doc = json.loads((out / "fit.json").read_text())
    ident = identity()
    if doc["identity"] != ident or doc["pipeline_id"] != token(ident):
        raise ValueError("Changed fit")
    nets = {}
    pars = {}
    for k in ARMS:
        path = out / f"{k}.npz"
        if sha(path) != doc["fits"][k]["weights"]:
            raise ValueError("Changed weights")
        net = network().eval()
        with np.load(path, allow_pickle=False) as weights, torch.no_grad():
            for j, layer in enumerate(m for m in net if hasattr(m, "weight")):
                layer.weight.copy_(torch.from_numpy(weights[f"w{j}"]))
                layer.bias.copy_(torch.from_numpy(weights[f"b{j}"]))
        nets[k] = net
        pars[k] = parameters(net)
    with Denoise(PTX / "search-support.ptx") as gpu, torch.inference_mode():
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            rec = doc["specimens"][str(i)]
            assert rec["raw"] == rs["sha256"]
            masks = {k: np.empty(SHAPE, "?") for k in ARMS}
            flips = dict.fromkeys(ARMS, 0)
            err = dict.fromkeys(ARMS, 0.0)
            baseline = np.empty(SHAPE, "?")
            for a in range(*Z, 2):
                x = features(raw, i, a, a + 2, gpu)
                prior = x[:, -1, 8:-8, 8:-8]
                at = slice(a - Z[0], a - Z[0] + 2)
                baseline[at] = prior >= 0
                for k in ARMS:
                    v = arm_input(x, k)
                    native = forward(v, pars[k], PTX / "spatial.ptx")[:, 0] + prior
                    cpu = nets[k](torch.from_numpy(v)).numpy()[:, 0] + prior
                    if not np.isfinite(native).all() or not np.isfinite(cpu).all():
                        raise ValueError("Invalid residual")
                    masks[k][at] = native >= 0
                    flips[k] += int(((cpu >= 0) != (native >= 0)).sum())
                    err[k] = max(err[k], float(np.max(abs(cpu - native))))
            rec["masks"] = {}
            rec["fixed_baseline"] = digest(baseline)
            for k in ARMS:
                path = out / f"sample{i}-{k}.npz"
                np.savez_compressed(path, mask=masks[k])
                rec["masks"][k] = dict(
                    file=sha(path),
                    mask=digest(masks[k]),
                    cpu_gpu_decision_disagreements=flips[k],
                    max_abs_cpu_gpu=err[k],
                )
            print(i, rec["masks"], flush=True)
    doc.update(origin_xyz=[360, 376, Z[0]], shape_zyx=list(SHAPE))
    write(out / "predictions.json", doc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["fit", "infer"])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (fit if a.phase == "fit" else infer)(a.cache, a.out)
