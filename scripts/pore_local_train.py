"""Train signed critical-voxel supervision on the sealed calibration tile census."""

import argparse
import hashlib
import json
import shutil
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
from pore_census_train import N, augment, schedule
from pore_census_train import identity as census_identity
from pore_local_connectivity import critical

OLD = ROOT / "pores/evidence/census-train"
EVID = ROOT / "pores/evidence/local-connectivity"
ARMS = ("census", "critical")


def identity():
    names = [
        "pores/LOCAL-CONNECTIVITY-PILOT.md",
        "scripts/pore_local_connectivity.py",
        "scripts/pore_local_train.py",
        "scripts/pore_local_predict.py",
        "scripts/pore_local_score.py",
    ]
    return dict(
        parent=census_identity(),
        files={
            n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in names
        },
        calibration_census=sha(EVID / "census.json"),
        parent_prediction=sha(OLD / "predictions.json"),
    )


def fit(cache, feature_dir, out):
    import torch
    from torch.nn import functional as fn

    configure()
    ident = identity()
    parent = json.loads((OLD / "fit.json").read_text())
    counts = json.loads((EVID / "census.json").read_text())
    if not counts["eligible"] or parent["identity"] != ident["parent"]:
        raise ValueError("Calibration or registered parent gate failed")
    out.mkdir(parents=True, exist_ok=False)
    x, t, c, exposure = {}, {}, {}, {}
    for i in DEV:
        ref = parent["specimens"][str(i)]
        f = feature_dir / f"cal-{i}.npy"
        _, rs = verified(cache, i, "raw", False)
        seg, ss = verified(cache, i, "segmented", False)
        t[i] = stack(seg, i, "segmented", 160, 184)
        x[i] = np.load(f, mmap_mode="r", allow_pickle=False)
        c[i] = critical(t[i])
        row = counts["specimens"][str(i)]
        if (
            [sha(f), digest(t[i]), rs["sha256"], ss["sha256"]]
            != [ref[k] for k in ("cal_features", "cal_mask", "raw", "segmented")]
            or digest(c[i]) != row["critical_sha256"]
            or not row["eligible"]
        ):
            raise ValueError("Source or signed center set changed")
        exposure[i] = [0, 0]
    rows, coverage = schedule({i: pools(t[i]) for i in DEV})
    cov = {str(i): v for i, v in coverage["census"].items()}
    if cov != parent["coverage"]["census"]:
        raise ValueError("Control schedule differs")
    ctrl = parent["fits"]["census"]
    path = OLD / "census.npz"
    if sha(path) != ctrl["weights"]:
        raise ValueError("Control weights differ")
    shutil.copyfile(path, out / "census.npz")
    torch.manual_seed(SEED)
    net = network()
    init = hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for p in net.parameters())
    ).hexdigest()
    if init != ctrl["init"]:
        raise ValueError("Different initialization")
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N - 1, eta_min=0.0001)
    trace = []
    for j in range(N):
        xs, ts, cs = [], [], []
        for i in DEV:
            for z, y, u, a in rows["census"][i][j]:
                xs.append(augment(x[i][z, :, y : y + 48, u : u + 48], a))
                b = augment(t[i][z, y : y + 32, u : u + 32], a)
                m = augment(c[i][z, y : y + 32, u : u + 32], a)
                ts.append(b)
                cs.append(m)
                for s in (0, 1):
                    exposure[i][s] += int((m & (b == s)).sum())
        y = torch.from_numpy(np.stack(ts).astype("f4")[:, None])
        m = torch.from_numpy(np.stack(cs)[:, None])
        v = region_logits(net, torch.from_numpy(np.stack(xs)))
        loss = (
            fn.binary_cross_entropy_with_logits(v, y)
            + 0.1 * ((v.sigmoid() - y).sum((2, 3)).abs() / (16 + y.sum((2, 3)))).mean()
        )
        terms = []
        for s in (0, 1):
            keep = m & (y == s)
            if torch.any(keep):
                terms.append(fn.binary_cross_entropy_with_logits(v[keep], y[keep]))
        added = torch.stack(terms).mean() if terms else v.new_zeros(())
        loss = loss + 0.10 * added
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if j % 256 == 0 or j == N - 1:
            trace.append([j, float(loss.detach()), float(added.detach())])
            print("critical", trace[-1], flush=True)
    wp = out / "critical.npz"
    np.savez(
        wp,
        **{
            f"{s}{j}": v
            for j, (w, b, _) in enumerate(parameters(net))
            for s, v in [("w", w), ("b", b)]
        },
    )
    write(
        out / "fit.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            specimens=parent["specimens"],
            coverage=cov,
            exposures_gap_connector=exposure,
            fits={"census": ctrl, "critical": dict(init=init, weights=sha(wp), trace=trace)},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for s in ("cache", "feature-dir", "out"):
        p.add_argument("--" + s, required=True, type=Path)
    a = p.parse_args()
    fit(a.cache, a.feature_dir, a.out)
