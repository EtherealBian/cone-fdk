"""Train and infer a 5-plane dense decoder; supervised reads are calibration-only."""

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
    SEED,
    SETTINGS,
    SHAPE,
    Denoise,
    Z,
    configure,
    digest,
    forward,
    parameters,
    pools,
    sha,
    stack,
    token,
    verified,
    write,
)
from pore_axial_replay import cpu
from pore_axial_score import read_mask
from pore_census_train import EVID, N, augment, schedule
from pore_census_train import identity as parent_identity
from scipy import ndimage as ndi

ARMS = ("slice", "volume")
FIELDS = (0, 5, 6, 7, 8, 9)
E = ROOT / "pores/evidence/five-plane"


def ident():
    return dict(
        parent=parent_identity(),
        census=sha(EVID / "predictions.json"),
        files={
            s: hashlib.sha256((ROOT / s).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for s in (
                "pores/FIVE-PLANE-PILOT.md",
                "scripts/pore_five_plane.py",
                "scripts/pore_five_score.py",
            )
        },
    )


def fields(path, i, a, b, den):
    from local_model import context_stack

    if i not in DEV or not (158 <= a < b <= 186 or 254 <= a < b <= 354):
        raise ValueError("Development halo only")
    scale, shift, ct = SETTINGS[i]
    raw = context_stack(path, i, a, b, 26)
    med = ndi.median_filter(raw, 3, mode="nearest")[1:-1]
    img = (den(med, 30 / scale) + shift) * scale
    byte = np.clip(np.floor(img + 0.5), 0, 255).astype("i2")
    y, x = np.ogrid[-5:6, -5:6]
    disk = (x * x + y * y <= 26)[None]
    lo = ndi.minimum_filter(byte, footprint=disk)
    hi = ndi.maximum_filter(byte, footprint=disk)
    mid = (hi + lo) // 2
    span = hi - lo

    def mapped(v):
        return ((v.astype("f4") + shift) * scale - 128) / 48

    prior = np.clip(np.where(span < ct, 127.5 - mid, mid - byte - 0.5), -4, 4)
    return np.ascontiguousarray(
        np.stack(
            [
                mapped(raw[1:-1]),
                mapped(med),
                (img - 128) / 48,
                (span - ct) / 6,
                (mid - byte) / 4,
                prior,
            ],
            axis=1,
        )[:, :, 18:-18, 18:-18],
        dtype="f4",
    )


def context(v, arm):
    if arm not in ARMS or v.ndim != 4 or v.shape[:2] != (5, 6):
        raise ValueError("Expected five native planes and six features")
    c = v[2]
    t = np.empty((5, *c.shape), dtype="f4")
    t[0] = c
    if arm == "slice":
        t[1:] = 0
    else:
        for k, j in enumerate((0, 1, 3, 4), 1):
            np.subtract(v[j], c, out=t[k])
    return t.reshape(30, *c.shape[1:])


def network():
    from torch import nn

    seq = []
    for j, d in enumerate(DILATIONS):
        last = j == len(DILATIONS) - 1
        layer = nn.Conv2d(30 if j == 0 else 16, 1 if last else 16, 1 if last else 3, dilation=d)
        seq.append(layer)
        if last:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        else:
            seq.append(nn.ReLU())
    return nn.Sequential(*seq)


def fit(cache, feature_dir, out):
    import torch
    from torch.nn import functional as fn

    configure()
    identity = ident()
    parent = json.loads((EVID / "predictions.json").read_bytes())
    out.mkdir(parents=True, exist_ok=False)
    xx = {}
    tt = {}
    specimens = {}
    with Denoise(PTX / "search-support.ptx") as den:
        for i in DEV:
            lock = parent["specimens"][str(i)]
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            t = stack(seg, i, "segmented", 160, 184)
            f = feature_dir / f"cal-{i}.npy"
            if [rs["sha256"], ss["sha256"], digest(t), sha(f)] != [
                lock[k] for k in ("raw", "segmented", "cal_mask", "cal_features")
            ]:
                raise ValueError("Changed source calibration")
            data = np.concatenate([fields(raw, i, z, z + 4, den) for z in range(158, 186, 4)])
            old = np.load(f, mmap_mode="r")
            assert all(np.array_equal(data[2:-2, j], old[:, k]) for j, k in enumerate(FIELDS))
            xx[i] = data
            tt[i] = t
            specimens[i] = dict(
                raw=rs["sha256"],
                segmented=ss["sha256"],
                cal_mask=digest(t),
                fields=digest(data),
                cal_overlap_voxels_per_field=24 * 272**2,
            )
    rows, cov = schedule({i: pools(tt[i]) for i in DEV})
    cov = {str(i): r for i, r in cov["census"].items()}
    assert cov == parent["coverage"]["census"]
    fits = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        net = network()
        init = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in net.parameters())
        ).hexdigest()
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N - 1, eta_min=0.0001)
        trace = []
        for j in range(N):
            inp = []
            targets = []
            for i in DEV:
                for z, y, x, flag in rows["census"][i][j]:
                    inp.append(
                        augment(context(xx[i][z : z + 5, :, y : y + 48, x : x + 48], arm), flag)
                    )
                    targets.append(augment(tt[i][z, y : y + 32, x : x + 32], flag))
            x = torch.from_numpy(np.stack(inp))
            t = torch.from_numpy(np.stack(targets).astype("f4")[:, None])
            v = net(x) + x[:, 5:6, 8:-8, 8:-8]
            loss = (
                fn.binary_cross_entropy_with_logits(v, t)
                + 0.1 * ((v.sigmoid() - t).sum((2, 3)).abs() / (16 + t.sum((2, 3)))).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            if j % 256 == 0 or j == N - 1:
                trace.append([j, float(loss.detach())])
                print(arm, trace[-1], flush=True)
        f = out / f"{arm}.npz"
        np.savez(
            f,
            **{
                f"{s}{j}": v
                for j, (w, b, _) in enumerate(parameters(net))
                for s, v in [("w", w), ("b", b)]
            },
        )
        fits[arm] = dict(init=init, weights=sha(f), trace=trace)
    assert fits["slice"]["init"] == fits["volume"]["init"]
    write(
        out / "fit.json",
        dict(
            identity=identity,
            pipeline_id=token(identity),
            specimens=specimens,
            fits=fits,
            coverage=cov,
        ),
    )


def predict(cache, out):
    configure()
    d = json.loads((out / "fit.json").read_bytes())
    assert ident() == d["identity"] and token(ident()) == d["pipeline_id"]
    pars = {}
    parent = json.loads((EVID / "predictions.json").read_bytes())
    for arm in ARMS:
        f = out / f"{arm}.npz"
        assert sha(f) == d["fits"][arm]["weights"]
        with np.load(f, allow_pickle=False) as a:
            pars[arm] = [(a[f"w{j}"], a[f"b{j}"], k) for j, k in enumerate(DILATIONS)]
    for i in DEV:
        r = d["specimens"][str(i)]
        raw, rs = verified(cache, i, "raw", False)
        assert r["raw"] == rs["sha256"]
        cr = parent["specimens"][str(i)]["masks"]["census"]
        base = read_mask(EVID, i, "census", cr)
        r["census_mask"] = digest(base)
        del base
        m = {k: np.empty(SHAPE, bool) for k in ARMS}
        first = dict.fromkeys(ARMS)
        counts = dict.fromkeys(ARMS, 0)
        maxima = dict.fromkeys(ARMS, 0.0)
        hf = hashlib.sha256()
        with Denoise(PTX / "search-support.ptx") as den:
            keep = fields(raw, i, 254, 258, den)
            hf.update(keep.tobytes())
            for z in range(*Z, 4):
                new = fields(raw, i, z + 2, z + 6, den)
                hf.update(new.tobytes())
                v = np.concatenate((keep, new))
                keep = new
                for u in range(4):
                    for k in ARMS:
                        inp = context(v[u : u + 5], k)
                        prior = inp[5, 8:-8, 8:-8]
                        native = forward(inp[None], pars[k], PTX / "spatial.ptx")[0, 0] + prior
                        independent = cpu(inp, pars[k])[0] + prior
                        assert np.isfinite(native).all() and np.isfinite(independent).all()
                        mask = native >= 0
                        m[k][z + u - Z[0]] = mask
                        neq = mask != (independent >= 0)
                        counts[k] += int(neq.sum())
                        maxima[k] = max(maxima[k], float(np.abs(native - independent).max()))
                        if neq.any() and first[k] is None:
                            y, x = np.argwhere(neq)[0]
                            first[k] = [
                                int(x) + 360,
                                int(y) + 376,
                                z + u,
                                float(native[y, x]),
                                float(independent[y, x]),
                            ]
                if (z - Z[0]) % 32 == 0:
                    print("predict", i, z, flush=True)
        r["transfer_fields"] = hf.hexdigest()
        r["masks"] = {}
        for k in ARMS:
            f = out / f"sample{i}-{k}.npz"
            with f.open("xb") as fp:
                np.savez_compressed(fp, mask=m[k])
            r["masks"][k] = dict(
                file=sha(f),
                mask=digest(m[k]),
                max_abs_cpu_gpu=maxima[k],
                cpu_gpu_decision_disagreements=counts[k],
                first_difference=first[k],
            )
            print(i, k, r["masks"][k], flush=True)
    d.update(origin_xyz=[360, 376, Z[0]], shape_zyx=list(SHAPE))
    write(out / "predictions.json", d)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("fit", "predict"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--feature-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fit(a.cache, a.feature_dir, a.out) if a.phase == "fit" else predict(a.cache, a.out)
