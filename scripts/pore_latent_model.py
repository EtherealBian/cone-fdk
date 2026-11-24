"""Candidate G: constrained intensity reconstruction with the registered digital Bernsen rule."""

import numpy as np
from pore_spatial_model import CHANNELS, SEED, batch, pools
from scipy import ndimage as ndi

RADII = (1, 3, 4, 4, 5, 5, 5, 4, 4, 3, 1)  # floor(sqrt(26-dy²)), dy=-5..5


def cpu_decision(base, correction, ct):
    a = np.clip(np.floor(base + 8 * np.tanh(correction / 8) + 0.5), 0, 255).astype("i2")
    y, x = np.ogrid[-5:6, -5:6]
    foot = (x * x + y * y <= 26)[None]
    lo = ndi.minimum_filter(a, footprint=foot, mode="nearest")
    hi = ndi.maximum_filter(a, footprint=foot, mode="nearest")
    mid = (lo + hi) // 2
    return np.where(hi - lo < ct, 127.5 - mid, mid - a - 0.5).astype("f4"), a


def cuda_decision(gpu, base, correction, ct):
    if base.ndim != 3 or base.shape != correction.shape or not 0 < ct < 256:
        raise ValueError("Unexpected intensity field or contrast")
    return np.split(gpu(np.concatenate([base, correction], axis=2), ct), 2, axis=2)


def network():
    from torch import nn

    layers = []
    for n in range(3):
        layers.extend((nn.Conv2d(CHANNELS if n == 0 else 12, 12, 3), nn.ReLU()))
    head = nn.Conv2d(12, 1, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)


def extrema(u):
    import torch
    from torch.nn import functional as fn

    # Max and min share one horizontal pool; only six distinct widths are needed.
    pair = torch.cat((u, -u), dim=1)
    rows = []
    h, w = u.shape[-2:]
    cache = {r: fn.max_pool2d(pair, (1, 2 * r + 1), stride=1) for r in set(RADII)}
    for j, r in enumerate(RADII):
        rows.append(cache[r][:, :, j : j + h - 10, 5 - r : 5 - r + w - 10])
    both = torch.stack(rows).amax(0)
    return -both[:, 1:], both[:, :1]


def floor_ste(x):
    return x + (x.floor() - x).detach()


def loss_fn(net, xx, tt, ct):
    import torch
    from torch.nn import functional as fn

    corr = 8 * (net(xx) / 8).tanh()
    base = xx[:, 3:4, 3:-3, 3:-3] * 48 + 128
    u = floor_ste((base + corr + 0.5).clamp(0, 255))
    lo, hi = extrema(u)
    mid = floor_ste((lo + hi) / 2)
    c = hi - lo - ct + 0.5
    h = (mid - u[:, :, 5:-5, 5:-5] - 0.5) / 0.8
    low = 127.5 - mid
    logc, lognc = fn.logsigmoid(c), fn.logsigmoid(-c)
    lp = torch.logaddexp(logc + fn.logsigmoid(h), lognc + fn.logsigmoid(low))
    ln = torch.logaddexp(logc + fn.logsigmoid(-h), lognc + fn.logsigmoid(-low))
    return (
        -(tt * lp + (1 - tt) * ln).mean()
        + 0.1 * ((lp.exp() - tt).sum((2, 3)).abs() / (16 + tt.sum((2, 3)))).mean()
        + 0.002 * corr.square().mean()
    )


def train(x, t, ct):
    import torch

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(3)
    rng = np.random.default_rng(SEED)
    net = network()
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1199, eta_min=0.0001)
    sets = pools(t)
    trace = []
    for j in range(1200):
        xx, tt = batch(x, t, rng, sets, j)
        loss = loss_fn(net, torch.from_numpy(xx), torch.from_numpy(tt), ct)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if j % 200 == 0 or j == 1199:
            trace.append([j, float(loss.detach())])
            print("G update", *trace[-1], flush=True)
    return net.eval(), trace
