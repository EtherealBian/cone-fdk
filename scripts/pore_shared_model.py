"""Shared F architecture, with a per-object relative mass loss on complete calibration pores."""

import numpy as np
from pore_spatial_model import HALO, SEED, network, pools
from scipy import ndimage as ndi

DEV = (2, 3, 5)


def owned_objects(t):
    if t.ndim != 3 or t.dtype != np.bool_ or not t.any():
        raise ValueError("Expected a nonempty binary calibration stack")
    lab, _ = ndi.label(t)
    cnt = np.bincount(lab.ravel())
    distance, nearest = ndi.distance_transform_edt(~t, return_indices=True)
    owner = lab[tuple(nearest)]
    records = []
    for j, box in enumerate(ndi.find_objects(lab), 1):
        if not 16 <= cnt[j] <= 511 or any(
            s.start < 2 or s.stop + 2 > n for s, n in zip(box, t.shape, strict=True)
        ):
            continue
        region = tuple(slice(s.start - 2, s.stop + 2) for s in box)
        mask = (owner[region] == j) & (distance[region] <= 2)
        n = int(cnt[j])
        if np.count_nonzero(t[region] & mask) != n:
            raise ValueError("Incomplete or contaminated ownership")
        records.append((j, region, mask, n))
    return records


def tiles(features, targets, sets, step, rng):
    xs, ts = [], []
    for i in DEV:
        x, t = features[i], targets[i]
        for k in range(2):
            mode = (2 * step + k + DEV.index(i)) % 3
            if mode == 0:
                z, y, u = [int(rng.integers(n)) for n in t.shape]
            else:
                pool = sets[i][mode - 1]
                z, y, u = pool[rng.integers(len(pool))]
            y, u = (int(np.clip(n - 8, 0, 240)) for n in (y, u))
            a, b = x[z, :, y : y + 32, u : u + 32], t[z, y : y + 16, u : u + 16]
            if rng.integers(2):
                a, b = a[:, ::-1], b[::-1]
            if rng.integers(2):
                a, b = a[:, :, ::-1], b[:, ::-1]
            if rng.integers(2):
                a, b = a.transpose(0, 2, 1), b.T
            xs.append(a)
            ts.append(b)
    return np.stack(xs), np.stack(ts)[:, None].astype("f4")


def region_logits(net, arr):
    return net(arr) + arr[:, -1:, HALO:-HALO, HALO:-HALO]


def object_input(x, rec):
    _, (z, y, u), mask, n = rec
    xx = x[z, :, y.start : y.stop + 2 * HALO, u.start : u.stop + 2 * HALO]
    if (xx.shape[0], xx.shape[2] - 2 * HALO, xx.shape[3] - 2 * HALO) != mask.shape:
        raise ValueError("Truncated context")
    return xx, mask, n


def train(x, t):
    import torch
    from torch.nn import functional as fn

    if set(x) != set(DEV) or set(t) != set(DEV):
        raise ValueError("Requires exactly the development calibration set")
    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(SEED)
    net = network()
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1799, eta_min=0.0001)
    sets = {i: pools(t[i]) for i in DEV}
    objects = {i: owned_objects(t[i]) for i in DEV}
    donors = [i for i in DEV if objects[i]]
    if not donors:
        raise ValueError("No complete calibration objects")
    census = {
        i: {
            f"{lo}-{hi}": sum(lo <= r[3] <= hi for r in objects[i])
            for lo, hi in [(16, 63), (64, 127), (128, 255), (256, 511)]
        }
        for i in DEV
    }
    draws = {i: 0 for i in DEV}
    trace = []
    for step in range(1800):
        xx, tt = [torch.from_numpy(v) for v in tiles(x, t, sets, step, rng)]
        v = region_logits(net, xx)
        loss = (
            fn.binary_cross_entropy_with_logits(v, tt)
            + 0.1 * ((v.sigmoid() - tt).sum((2, 3)).abs() / (16 + tt.sum((2, 3)))).mean()
        )
        obj = None
        if step % 4 == 0:
            i = donors[(step // 4) % len(donors)]
            rec = objects[i][rng.integers(len(objects[i]))]
            xx, mask, n = object_input(x[i], rec)
            vv = region_logits(net, torch.from_numpy(xx))[:, 0]
            mass = vv.sigmoid()[torch.from_numpy(mask)].sum()
            oloss = (mass - n).abs() / n
            loss = loss + 0.5 * oloss
            obj = dict(
                specimen=i,
                calibration_id=rec[0],
                size=n,
                relative_soft_error=float(oloss.detach()),
            )
            draws[i] += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1799:
            trace.append(dict(step=step, loss=float(loss.detach()), object=obj))
            print("H update", trace[-1], flush=True)
    net.eval()
    return net, dict(object_census=census, object_draws=draws, trace=trace)
