"""A full calibration tile census versus equal-cost uniform sampling with replacement."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import (
    DEV,
    ROOT,
    SEED,
    configure,
    digest,
    network,
    parameters,
    pools,
    region_logits,
    sha,
    stack,
    token,
    verified,
    write,
)
from pore_axial_raw import identity as axial_identity

ARMS = ("replacement", "census")
N = 24 * 8 * 8
EVID = ROOT / "pores/evidence/census-train"


def identity():
    return dict(
        parent=axial_identity(),
        files={
            s: hashlib.sha256((ROOT / s).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for s in (
                "scripts/pore_census_train.py",
                "scripts/pore_census_predict.py",
                "pores/CENSUS-TRAIN-PILOT.md",
                "scripts/pore_axial_replay.py",
            )
        },
    )


def schedule(sets):
    schedules = {k: {} for k in ARMS}
    coverage = {k: {} for k in ARMS}
    for i in DEV:
        independent = np.random.default_rng(np.random.SeedSequence([SEED, i, 0]))
        grid = {
            "census": independent.permutation(N),
            "replacement": np.random.default_rng([SEED, i, 1]).integers(N, size=N),
        }
        common = np.random.default_rng([SEED, i, 2])
        extras = []
        for j in range(N):
            pool = sets[i][j % 2]
            z, y, x = pool[common.integers(len(pool))]
            extras.append([z, np.clip(y - 16, 0, 224), np.clip(x - 16, 0, 224)])
        transforms = common.integers(8, size=(N, 2))
        for k in ARMS:
            v = grid[k]
            coords = np.stack((v // 64, v // 8 % 8 * 32, v % 8 * 32), axis=1)
            rows = np.concatenate(
                (np.stack((coords, extras), axis=1), transforms[:, :, None]), axis=2
            ).astype("i4")
            seen = np.zeros((24, 256, 256), bool)
            for z, y, x, _ in rows.reshape(-1, 4):
                seen[z, y : y + 32, x : x + 32] = True
            coverage[k][i] = dict(
                unique_grid_tiles=len(np.unique(v)),
                unique_voxels=int(seen.sum()),
                total=seen.size,
                schedule_sha256=digest(rows),
                common_sha256=digest(rows[:, 1]),
                augment_sha256=digest(rows[:, :, 3]),
            )
            if k == "census" and not seen.all():
                raise ValueError("Incomplete census")
            schedules[k][i] = rows
    return schedules, coverage


def augment(a, flag):
    if flag & 1:
        a = a[..., ::-1, :]
    if flag & 2:
        a = a[..., ::-1]
    if flag & 4:
        a = a.swapaxes(-2, -1)
    return a


def fit(cache, feature_dir, out):
    import torch
    from torch.nn import functional as fn

    out.mkdir(parents=True, exist_ok=False)
    configure()
    ident = identity()
    parent = json.loads((ROOT / "pores/evidence/axial-raw/predictions.json").read_text())
    x = {}
    t = {}
    rec = {}
    if ident["parent"] != parent["identity"]:
        raise ValueError("Unregistered parent")
    for i in DEV:
        ref = parent["specimens"][str(i)]
        raw, rs = verified(cache, i, "raw", False)
        seg, ss = verified(cache, i, "segmented", False)
        f = feature_dir / f"cal-{i}.npy"
        x[i] = np.load(f, mmap_mode="r", allow_pickle=False)
        t[i] = stack(seg, i, "segmented", 160, 184)
        rec[i] = dict(
            raw=rs["sha256"], segmented=ss["sha256"], cal_mask=digest(t[i]), cal_features=sha(f)
        )
        if (
            any(ref[a] != v for a, v in rec[i].items())
            or x[i].shape != (24, 10, 272, 272)
            or x[i].dtype != np.float32
        ):
            raise ValueError("Unverified calibration cache")
    schedules, coverage = schedule({i: pools(t[i]) for i in DEV})
    fits = {}
    for k in ARMS:
        torch.manual_seed(SEED)
        net = network()
        init = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in net.parameters())
        ).hexdigest()
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N - 1, eta_min=0.0001)
        trace = []
        for j in range(N):
            inputs = []
            targets = []
            for i in DEV:
                for z, y, u, a in schedules[k][i][j]:
                    inputs.append(augment(x[i][z, :, y : y + 48, u : u + 48], a))
                    targets.append(augment(t[i][z, y : y + 32, u : u + 32], a))
            y = torch.from_numpy(np.stack(targets).astype("f4")[:, None])
            v = region_logits(net, torch.from_numpy(np.stack(inputs)))
            loss = (
                fn.binary_cross_entropy_with_logits(v, y)
                + 0.1 * ((v.sigmoid() - y).sum((2, 3)).abs() / (16 + y.sum((2, 3)))).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if j % 256 == 0 or j == N - 1:
                trace.append([j, float(loss.detach())])
                print(k, trace[-1], flush=True)
        wp = out / f"{k}.npz"
        np.savez(
            wp,
            **{
                f"{s}{j}": a
                for j, (w, b, _) in enumerate(parameters(net))
                for s, a in [("w", w), ("b", b)]
            },
        )
        fits[k] = dict(init=init, weights=sha(wp), trace=trace)
    if fits[ARMS[0]]["init"] != fits[ARMS[1]]["init"]:
        raise ValueError("Different initializations")
    write(
        out / "fit.json",
        dict(
            identity=ident, pipeline_id=token(ident), fits=fits, coverage=coverage, specimens=rec
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "feature-dir", "out"):
        ap.add_argument("--" + name, required=True, type=Path)
    a = ap.parse_args()
    fit(a.cache, a.feature_dir, a.out)
