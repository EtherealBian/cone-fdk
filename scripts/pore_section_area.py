"""Two matched learners; only the auxiliary planar-area loss differs."""

from collections import Counter, defaultdict

import numpy as np
from pore_shared_model import DEV, region_logits, tiles
from pore_spatial_model import HALO, SEED, network, pools
from scipy import ndimage as ndi


def sections(t):
    if t.ndim != 3 or t.dtype != np.bool_:
        raise ValueError("Expected binary calibration stack")
    parents, _ = ndi.label(t)
    boundary = set(
        np.unique(
            np.concatenate(
                [np.take(parents, z, axis=a).ravel() for a in range(3) for z in (0, -1)]
            )
        ).tolist()
    )
    zface = set(np.unique(parents[[0, -1]]).tolist())
    out = defaultdict(list)
    for z, plane in enumerate(t):
        lab, _ = ndi.label(plane)
        if not plane.any():
            continue
        area = np.bincount(lab.ravel())
        dist, near = ndi.distance_transform_edt(~plane, return_indices=True)
        owners = lab[tuple(near)]
        for j, box in enumerate(ndi.find_objects(lab), 1):
            n = int(area[j])
            if not 4 <= n <= 255 or any(
                s.start < 2 or s.stop + 2 > d for s, d in zip(box, plane.shape, strict=True)
            ):
                continue
            reg = tuple(slice(s.start - 2, s.stop + 2) for s in box)
            fg = lab[reg] == j
            ids = np.unique(parents[z][reg][fg])
            mask = (owners[reg] == j) & (dist[reg] <= 2)
            if len(ids) != 1 or not ids[0] or not np.array_equal(plane[reg] & mask, fg):
                raise ValueError("Incorrect ownership")
            out[int(ids[0])].append((z, j, *reg, mask, n))
    details = {
        str(k): {
            "touches_face": k in boundary,
            "touches_z": k in zface,
            "sections": [[r[0], r[1], r[-1]] for r in rows],
        }
        for k, rows in out.items()
    }
    return dict(sorted(out.items())), details


def draw(groups, rng):
    keys = list(groups)
    k = keys[rng.integers(len(keys))]
    return k, groups[k][rng.integers(len(groups[k]))]


def train(x, t, out):
    import hashlib

    import torch
    from pore_spatial_model import parameters
    from torch.nn import functional as fn

    if set(x) != set(DEV) or set(t) != set(DEV):
        raise ValueError("Only the complete development calibration set is admissible")
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    sets = {i: pools(t[i]) for i in DEV}
    g = {i: sections(t[i]) for i in DEV}
    if any(not g[i][0] for i in DEV):
        raise ValueError("Unsupported specimen: no eligible parent")
    records = {"parents": {i: g[i][1] for i in DEV}, "arms": {}}
    nets = {}
    for name in ("control", "area"):
        torch.manual_seed(SEED)
        rng = np.random.default_rng(SEED)
        aux = np.random.default_rng(SEED + 1)
        net = network()
        opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1799, eta_min=0.0001)
        h = hashlib.sha256()
        initial = hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in net.parameters())
        ).hexdigest()
        visits = Counter()
        trace = []
        for step in range(1800):
            xx, tt = tiles(x, t, sets, step, rng)
            h.update(xx.tobytes())
            h.update(tt.tobytes())
            xx, tt = map(torch.from_numpy, (xx, tt))
            v = region_logits(net, xx)
            loss = (
                fn.binary_cross_entropy_with_logits(v, tt)
                + 0.1 * ((v.sigmoid() - tt).sum((2, 3)).abs() / (16 + tt.sum((2, 3)))).mean()
            )
            if name == "area" and step % 4 == 0:
                i = DEV[(step // 4) % 3]
                p, (z, j, y, u, mask, n) = draw(g[i][0], aux)
                xx = np.array(
                    x[i][z : z + 1, :, y.start : y.stop + 2 * HALO, u.start : u.stop + 2 * HALO],
                    copy=True,
                )
                vv = region_logits(net, torch.from_numpy(xx))[0, 0]
                loss = loss + 0.5 * (vv.sigmoid()[torch.from_numpy(mask)].sum() - n).abs() / n
                visits[(i, p, z, j)] += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 300 == 0 or step == 1799:
                trace.append([step, float(loss.detach())])
                print(name, *trace[-1], flush=True)
        net.eval()
        nets[name] = net
        pars = parameters(net)
        np.savez(
            out / f"{name}.npz",
            **{f"{k}{j}": v for j, (w, b, _) in enumerate(pars) for k, v in [("w", w), ("b", b)]},
        )
        records["arms"][name] = dict(
            init=initial,
            patch_stream=h.hexdigest(),
            visits=[[*k, v] for k, v in sorted(visits.items())],
            trace=trace,
        )
    if any(
        records["arms"]["area"][k] != records["arms"]["control"][k]
        for k in ("init", "patch_stream")
    ):
        raise ValueError("Control initialization or patch stream mismatch")
    return nets, records
