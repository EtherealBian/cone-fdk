"""Matched digital-interface supervision, same-band sign control and raw transfer inference."""

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
    SHAPE,
    Denoise,
    Z,
    configure,
    digest,
    features,
    forward,
    network,
    parameters,
    pools,
    region_logits,
    sha,
    token,
    verified,
    write,
)
from pore_axial_replay import cpu
from pore_census_train import EVID, N, augment, schedule
from pore_census_train import identity as parent_identity
from pore_distance_targets import calibration

E = ROOT / "pores/evidence/distance-target"
ARMS = ("sign", "distance")


def load(p):
    return json.loads(p.read_bytes())


def identity():
    files = (
        "pores/DISTANCE-TARGET-PILOT.md",
        "scripts/pore_distance_targets.py",
        "scripts/pore_distance.py",
        "scripts/pore_distance_score.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_section_score.py",
        "scripts/pore_transfer.py",
        "scripts/pore_axial_score.py",
        "pores/pore_check.py",
    )
    return dict(
        parent=parent_identity(),
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in files
        },
        native={k: sha(PTX / k) for k in ("cone-pores.exe", "pores.ptx")},
        census=sha(E / "census.json"),
        prior=sha(EVID / "predictions.json"),
    )


def fit(cache, feature_dir, out):
    import torch
    from torch.nn import functional as fn

    ident = identity()
    t, d, c = calibration(cache)
    if c != load(E / "census.json") or not c["eligible"]:
        raise ValueError("Calibration registration or support mismatch")
    par = load(EVID / "fit.json")
    assert ident["parent"] == par["identity"]
    configure()
    x = {}
    for i in DEV:
        f = feature_dir / f"cal-{i}.npy"
        ref = par["specimens"][str(i)]
        assert (
            sha(f) == ref["cal_features"]
            and verified(cache, i, "raw", False)[1]["sha256"] == ref["raw"]
        )
        x[i] = np.load(f, mmap_mode="r", allow_pickle=False)
        assert x[i].shape == (24, 10, 272, 272) and x[i].dtype == np.float32
    scheds, cov = schedule({i: pools(t[i]) for i in DEV})
    assert {str(i): cov["census"][i] for i in DEV} == par["coverage"]["census"]
    out.mkdir(parents=True, exist_ok=False)
    fits = {}
    for k in ARMS:
        torch.manual_seed(SEED)
        net = network()
        init = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in net.parameters())
        ).hexdigest()
        assert init == par["fits"]["census"]["init"]
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N - 1, eta_min=0.0001)
        stream = hashlib.sha256()
        trace = []
        centers = 0
        empty = 0
        for j in range(N):
            inputs = []
            targets = []
            aux = []
            for i in DEV:
                for z, y, u, a in scheds["census"][i][j]:
                    inputs.append(augment(x[i][z, :, y : y + 48, u : u + 48], a))
                    tile = np.s_[z, y : y + 32, u : u + 32]
                    targets.append(augment(t[i][tile], a))
                    aux.append(augment(d[i][tile], a))
            xx = np.stack(inputs)
            yy = np.stack(targets)
            dd = np.stack(aux)
            for arr in (xx, yy, dd):
                stream.update(np.ascontiguousarray(arr).view("u1"))
            y = torch.from_numpy(yy.astype("f4")[:, None])
            s = torch.from_numpy(dd[:, None])
            valid = s != 0
            n = int(valid.sum())
            centers += n
            empty += int(n == 0)
            v = region_logits(net, torch.from_numpy(xx))
            s = s if k == "distance" else 0.5 * s.sign()
            extension = (
                fn.smooth_l1_loss(v[valid] / 2, s[valid], beta=1.0) if n else v.new_zeros(())
            )
            base = (
                fn.binary_cross_entropy_with_logits(v, y)
                + 0.1 * ((v.sigmoid() - y).sum((2, 3)).abs() / (16 + y.sum((2, 3)))).mean()
            )
            loss = base + 0.25 * extension
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if j % 256 == 0 or j == N - 1:
                trace.append(
                    [j, float(base.detach()), float(extension.detach()), float(loss.detach())]
                )
                print(k, trace[-1], flush=True)
        wp = out / f"{k}.npz"
        with wp.open("xb") as fp:
            np.savez(
                fp,
                **{
                    f"{s}{j}": a
                    for j, (w, b, _) in enumerate(parameters(net))
                    for s, a in [("w", w), ("b", b)]
                },
            )
        fits[k] = dict(
            init=init,
            stream=stream.hexdigest(),
            weights=sha(wp),
            trace=trace,
            band_exposures=centers,
            empty_batches=empty,
        )
    assert all(
        fits["sign"][k] == fits["distance"][k]
        for k in ("init", "stream", "band_exposures", "empty_batches")
    )
    write(
        out / "fit.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            fits=fits,
            coverage=cov["census"],
            specimens=par["specimens"],
        ),
    )


def predict(cache, out):
    ident = identity()
    doc = load(out / "fit.json")
    assert ident == doc["identity"] and token(ident) == doc["pipeline_id"]
    pars = {}
    for k in ARMS:
        wp = out / f"{k}.npz"
        assert sha(wp) == doc["fits"][k]["weights"]
        with np.load(wp, allow_pickle=False) as f:
            pars[k] = [(f[f"w{j}"], f[f"b{j}"], v) for j, v in enumerate(DILATIONS)]
    for i in DEV:
        raw, rs = verified(cache, i, "raw", False)
        rec = doc["specimens"][str(i)]
        assert rs["sha256"] == rec["raw"]
        masks = {k: np.empty(SHAPE, bool) for k in ARMS}
        diff = dict.fromkeys(ARMS, 0)
        err = dict.fromkeys(ARMS, 0.0)
        first = dict.fromkeys(ARMS, None)
        h = hashlib.sha256()
        with Denoise(PTX / "search-support.ptx") as gpu:
            for off in range(0, SHAPE[0], 4):
                x = features(raw, i, Z[0] + off, Z[0] + off + 4, gpu)
                h.update(np.ascontiguousarray(x).view("u1"))
                prior = x[:, -1, 8:-8, 8:-8]
                for k in ARMS:
                    n = forward(x, pars[k], PTX / "spatial.ptx")[:, 0] + prior
                    c = np.stack([cpu(a, pars[k])[0] for a in x]) + prior
                    if not (np.isfinite(n).all() and np.isfinite(c).all()):
                        raise ValueError("Nonfinite prediction")
                    v = n >= 0
                    neq = v != (c >= 0)
                    diff[k] += int(neq.sum())
                    err[k] = max(err[k], float(abs(n - c).max()))
                    if first[k] is None and neq.any():
                        z, y, u = np.argwhere(neq)[0]
                        first[k] = [
                            int(u + 360),
                            int(y + 376),
                            int(z + Z[0] + off),
                            float(n[z, y, u]),
                            float(c[z, y, u]),
                        ]
                    masks[k][off : off + 4] = v
        rec["features_f32"] = h.hexdigest()
        rec["masks"] = {}
        for k, v in masks.items():
            f = out / f"sample{i}-{k}.npz"
            with f.open("xb") as fp:
                np.savez_compressed(fp, mask=v)
            rec["masks"][k] = dict(
                file=sha(f),
                mask=digest(v),
                cpu_gpu_differences=diff[k],
                max_error=err[k],
                first_difference=first[k],
            )
            print(i, k, rec["masks"][k], flush=True)
    doc.update(origin_xyz=[360, 376, Z[0]], shape_zyx=list(SHAPE))
    write(out / "predictions.json", doc)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "predict"))
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--feature-dir", type=Path)
    a = p.parse_args()
    if a.stage == "fit":
        if a.feature_dir is None:
            p.error("fit needs --feature-dir")
        fit(a.cache, a.feature_dir, a.out)
    else:
        predict(a.cache, a.out)
