"""Fit and seal matched native/coarse raw residuals without transfer labels."""

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
    parameters,
    pools,
    sha,
    stack,
    token,
    verified,
    write,
)
from pore_axial_replay import cpu
from pore_census_train import EVID as BASE
from pore_census_train import N, augment, schedule
from pore_census_train import identity as base_id
from pore_raw_grid import SCALES, GridGPU, cpu_grid, expand, scalar

from local_model import context_stack  # noqa: E402

E = ROOT / "pores/evidence/two-scale"


def identity():
    return dict(
        parent=base_id(),
        base=sha(BASE / "predictions.json"),
        grid_ptx=sha(PTX / "raw-grid.ptx"),
        files={
            n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in (
                "pores/TWO-SCALE-PILOT.md",
                "scripts/pore_raw_grid.cu",
                "scripts/pore_raw_grid.py",
                "scripts/pore_two_scale.py",
                "scripts/pore_two_scale_score.py",
            )
        },
    )


def weights(path, lock, ds):
    if sha(path) != lock:
        raise ValueError("Changed weights")
    with np.load(path, allow_pickle=False) as a:
        return [(a[f"w{j}"], a[f"b{j}"], d) for j, d in enumerate(ds)]


def baseline():
    b = json.loads((BASE / "predictions.json").read_bytes())
    pars = weights(BASE / "census.npz", b["fits"]["census"]["weights"], DILATIONS)
    if b["identity"] != base_id():
        raise ValueError("Wrong base code")
    return b, pars


def check_grids(vol, s, i, z, gpu, check_scalar):
    g = gpu(vol, s, i)
    c = cpu_grid(vol, s, i)
    err = float(abs(c - g).max())
    n = 0
    scalar_err = 0.0
    if not np.isfinite(err) or err > 1e-5:
        raise ValueError(("grid", i, z, s, err))
    if check_scalar:
        mat = expand(g, s)
        ny, nx = mat.shape[-2:]
        rng = np.random.default_rng([i, z, s, 61429])
        pts = {(0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1)}
        while len(pts) < 32:
            pts.add((int(rng.integers(ny)), int(rng.integers(nx))))
        for y, x in sorted(pts):
            scalar_err = max(float(abs(scalar(vol, s, i, y, x) - mat[:, y, x]).max()), scalar_err)
        if scalar_err > 1e-5:
            raise ValueError(("scalar grid", i, z, s, scalar_err))
        n = len(pts)
    return g, c, [err, n, scalar_err]


def fit(cache, feature_dir, out):
    import torch
    from torch import nn
    from torch.nn import functional as fn

    configure()
    ident = identity()
    ctrl, pars = baseline()
    out.mkdir(parents=True, exist_ok=False)
    t = {}
    xs = {}
    base = {}
    rec = {}
    audits = {}
    with GridGPU() as grid:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            parent = ctrl["specimens"][str(i)]
            path = feature_dir / f"cal-{i}.npy"
            t[i] = stack(seg, i, "segmented", 160, 184)
            if [rs["sha256"], ss["sha256"], digest(t[i]), sha(path)] != [
                parent[k] for k in ("raw", "segmented", "cal_mask", "cal_features")
            ]:
                raise ValueError("Changed calibration")
            axial = np.load(path, allow_pickle=False, mmap_mode="r")
            v = context_stack(raw, i, 160, 184, 16)
            base[i] = np.empty((24, 256, 256), "f4")
            xs[i] = {}
            audits[i] = {}
            for k, s in SCALES.items():
                g = []
                rows = []
                for j in range(24):
                    gg, _, r = check_grids(v[j : j + 3], s, i, 160 + j, grid, j in (0, 12, 23))
                    g.append(gg)
                    rows.append(r)
                xs[i][k] = np.stack(g)
                audits[i][k] = rows
            for j in range(24):
                base[i][j] = (
                    forward(axial[j : j + 1], pars, PTX / "spatial.ptx")[0, 0]
                    + axial[j, -1, 8:-8, 8:-8]
                )
            rec[i] = dict(
                raw=rs["sha256"],
                segmented=ss["sha256"],
                cal_mask=digest(t[i]),
                cal_features=sha(path),
                base_logits=digest(base[i]),
                grids={k: digest(v) for k, v in xs[i].items()},
            )
    rows, cov = schedule({i: pools(t[i]) for i in DEV})
    cov = {str(i): r for i, r in cov["census"].items()}
    if cov != ctrl["coverage"]["census"]:
        raise ValueError("Different tile coordinates")
    fits = {}
    for k, s in SCALES.items():
        torch.manual_seed(SEED)
        head = nn.Sequential(nn.Conv2d(27, 16, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        init = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in head.parameters())
        ).hexdigest()
        opt = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.0001)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N - 1, eta_min=0.0001)
        trace = []
        for step in range(N):
            inp = []
            prior = []
            targets = []
            for i in DEV:
                for z, y, x, a in rows["census"][i][step]:
                    sl = np.s_[z, y : y + 32, x : x + 32]
                    inp.append(augment(expand(xs[i][k][z, :, y : y + 48, x : x + 48], s), a))
                    prior.append(augment(base[i][sl], a))
                    targets.append(augment(t[i][sl], a))
            v = head(torch.from_numpy(np.stack(inp))) + torch.from_numpy(np.stack(prior)[:, None])
            y = torch.from_numpy(np.stack(targets).astype("f4")[:, None])
            loss = (
                fn.binary_cross_entropy_with_logits(v, y)
                + 0.1 * ((v.sigmoid() - y).sum((2, 3)).abs() / (16 + y.sum((2, 3)))).mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            if step % 256 == 0 or step == N - 1:
                trace.append([step, float(loss.detach())])
                print(k, trace[-1], flush=True)
        f = out / f"{k}.npz"
        np.savez(
            f,
            **{
                f"{n}{j}": v
                for j, (w, b, _) in enumerate(parameters(head))
                for n, v in [("w", w), ("b", b)]
            },
        )
        fits[k] = dict(init=init, weights=sha(f), trace=trace)
    if fits["fine"]["init"] != fits["coarse"]["init"]:
        raise ValueError("Unmatched heads")
    write(
        out / "fit.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            specimens=rec,
            fits=fits,
            coverage=cov,
            cal_grid_audits=audits,
        ),
    )


def infer(cache, out):
    configure()
    d = json.loads((out / "fit.json").read_bytes())
    ident = identity()
    ctrl, base = baseline()
    if d["identity"] != ident or d["pipeline_id"] != token(ident):
        raise ValueError("Wrong fit")
    heads = {k: weights(out / f"{k}.npz", r["weights"], (1, 1)) for k, r in d["fits"].items()}
    keys = ("census", *SCALES)
    for i in DEV:
        raw, rs = verified(cache, i, "raw", False)
        r = d["specimens"][str(i)]
        if rs["sha256"] != r["raw"]:
            raise ValueError("Different raw scan")
        masks = {k: np.empty(SHAPE, bool) for k in keys}
        counts = dict.fromkeys(keys, 0)
        maxima = dict.fromkeys(keys, 0.0)
        first = dict.fromkeys(keys, None)
        audits = {k: [] for k in SCALES}
        with Denoise(PTX / "search-support.ptx") as den, GridGPU() as grid:
            for z in range(*Z):
                a = features(raw, i, z, z + 1, den)
                pr = a[0, -1, 8:-8, 8:-8]
                v = context_stack(raw, i, z, z + 1, 16)
                n = forward(a, base, PTX / "spatial.ptx")[0, 0] + pr
                c = cpu(a[0], base)[0] + pr
                scores = {"census": (n, c)}
                for k, s in SCALES.items():
                    ng, cg, audit = check_grids(
                        v, s, i, z, grid, z in (Z[0], sum(Z) // 2, Z[1] - 1)
                    )
                    audits[k].append(audit)
                    scores[k] = (
                        n + forward(expand(ng, s)[None], heads[k], PTX / "spatial.ptx")[0, 0],
                        c + cpu(expand(cg, s), heads[k])[0],
                    )
                for k, (nv, cv) in scores.items():
                    if not np.isfinite(nv).all() or not np.isfinite(cv).all():
                        raise ValueError("Nonfinite logit")
                    m = nv >= 0
                    change = m != (cv >= 0)
                    j = z - Z[0]
                    masks[k][j] = m
                    counts[k] += int(change.sum())
                    maxima[k] = max(maxima[k], float(abs(nv - cv).max()))
                    if change.any() and first[k] is None:
                        y, x = np.argwhere(change)[0]
                        first[k] = [
                            int(x) + 360,
                            int(y) + 376,
                            z,
                            float(nv[y, x]),
                            float(cv[y, x]),
                        ]
                if (z - Z[0]) % 32 == 0:
                    print(i, z, flush=True)
        r["masks"] = {}
        r["grid_audit"] = audits
        for k in keys:
            m = masks[k]
            f = out / f"sample{i}-{k}.npz"
            if (
                k == "census"
                and digest(m) != ctrl["specimens"][str(i)]["masks"]["census"]["mask"]
            ):
                raise ValueError("Different census base")
            with f.open("xb") as fp:
                np.savez_compressed(fp, mask=m)
            r["masks"][k] = dict(
                file=sha(f),
                mask=digest(m),
                cpu_gpu_decision_disagreements=counts[k],
                max_abs_cpu_gpu=maxima[k],
                first_difference=first[k],
            )
            print(i, k, r["masks"][k], flush=True)
    d.update(origin_xyz=[360, 376, Z[0]], shape_zyx=list(SHAPE))
    write(out / "predictions.json", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("fit", "predict"))
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--feature-dir", type=Path)
    a = ap.parse_args()
    fit(a.cache, a.feature_dir, a.out) if a.phase == "fit" else infer(a.cache, a.out)
