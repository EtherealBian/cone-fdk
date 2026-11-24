"""Candidate F inputs, calibration sampler and per-scan residual plane network."""

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
from local_model import SETTINGS, median_stack  # noqa: E402
from pore_transfer import label  # noqa: E402

DILATIONS = (1, 2, 4, 1, 1)
CHANNELS = 22
HALO = 8
SOURCE_HALO = 26
SEED = 61429


def fields(raw, i, a, b, gpu):
    med = median_stack(raw, i, a - 1, b + 1, SOURCE_HALO)
    scale, offset, ct = SETTINGS[i]
    den = gpu(med, 30 / scale)
    h = SOURCE_HALO - HALO
    sl = (slice(1, -1), slice(h, -h), slice(h, -h))
    out = []
    prior = None
    yy, xx = np.ogrid[-5:6, -5:6]
    disk = (xx * xx + yy * yy <= 26)[None]
    for j in range(6):
        img = (den[:, j] + offset) * scale
        u = np.clip(np.floor(img + 0.5), 0, 255)
        lo = ndi.minimum_filter(u, footprint=disk)
        hi = ndi.maximum_filter(u, footprint=disk)
        mid = np.floor((lo + hi) * 0.5)
        span = hi - lo
        margin = mid - img
        out.extend(((img[sl] - 128) / 48, margin[sl] / 4, (span[sl] - ct) / 6))
        if j == 1:
            m = margin[:, h:-h, h:-h]
            adj = (m[:-2] + m[2:]) * 0.5
            diff = np.abs(m[:-2] - m[2:])
            prior = np.clip(np.where(span[sl] < ct, 128 - mid[sl], margin[sl]), -4, 4)
            zctx = [adj / 4, diff / 4]
    out.extend([((med[sl].astype("f4") + offset) * scale - 128) / 48, *zctx, prior])
    arr = np.stack(out, axis=1).astype("f4")
    np.clip(arr, -8, 8, out=arr)
    if arr.shape != (b - a, CHANNELS, 272, 272) or not np.isfinite(arr).all():
        raise ValueError("Invalid contextual input shape")
    return arr


def pools(truth):
    lab, n, whole, _ = label(truth)
    border = ndi.binary_dilation(truth) ^ ndi.binary_erosion(truth)
    near = ndi.distance_transform_edt(~truth, return_indices=True, return_distances=False)
    small = (whole & (n >= 16) & (n <= 255))[lab[tuple(near)]]
    qb = np.argwhere(border)
    qs = np.argwhere(border & small)
    if not len(qb):
        raise ValueError("No calibration pore transitions")
    return qb, qs if len(qs) else qb


def batch(x, t, rng, sets, step):
    xs = []
    ts = []
    for k in range(8):
        mode = (8 * step + k) % 3
        if mode == 0:
            z, y, u = [int(rng.integers(n)) for n in t.shape]
        else:
            pool = sets[mode - 1]
            z, y, u = pool[rng.integers(len(pool))]
        y = int(np.clip(y - 8, 0, 240))
        u = int(np.clip(u - 8, 0, 240))
        v = x[z, :, y : y + 32, u : u + 32]
        s = t[z, y : y + 16, u : u + 16]
        if rng.integers(2):
            v, s = v[:, ::-1, :], s[::-1, :]
        if rng.integers(2):
            v, s = v[:, :, ::-1], s[:, ::-1]
        if rng.integers(2):
            v, s = v.transpose(0, 2, 1), s.T
        xs.append(v)
        ts.append(s)
    return np.stack(xs), np.stack(ts).astype("f4")[:, None]


def network():
    from torch import nn

    seq = []
    ci = CHANNELS
    for j, d in enumerate(DILATIONS):
        last = j == 4
        layer = nn.Conv2d(ci, 1 if last else 16, 1 if last else 3, dilation=d)
        seq.append(layer)
        if last:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        else:
            seq.append(nn.ReLU())
            ci = 16
    return nn.Sequential(*seq)


def train(x, t):
    import torch
    from torch.nn import functional as fn

    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(SEED)
    net = network()
    opt = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1199, eta_min=0.0001)
    sets = pools(t)
    losses = []
    for step in range(1200):
        xx, tt = batch(x, t, rng, sets, step)
        xx = torch.from_numpy(xx)
        tt = torch.from_numpy(tt)
        out = net(xx) + xx[:, -1:, 8:-8, 8:-8]
        loss = (
            fn.binary_cross_entropy_with_logits(out, tt)
            + 0.1
            * ((out.sigmoid() - tt).sum(dim=(2, 3)).abs() / (16 + tt.sum(dim=(2, 3)))).mean()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1199:
            losses.append([step, float(loss.detach())])
            print("update", *losses[-1], flush=True)
    net.eval()
    return net, losses


def parameters(net):
    return [
        (layer.weight.detach().numpy(), layer.bias.detach().numpy(), layer.dilation[0])
        for layer in net
        if hasattr(layer, "weight")
    ]
