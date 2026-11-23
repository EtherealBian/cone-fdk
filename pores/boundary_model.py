"""Multi-scale boundary descriptors; only calibration blocks may choose the cutoff."""

import numpy as np
from local_model import SETTINGS
from residual_model import probability
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def features(median, i, gpu):
    scale, shift, ct = SETTINGS[i]

    def crop(a):
        return a[:, 24:-24, 24:-24]

    field = gpu(median, 30 / scale)
    group = []
    keep = None
    yy, xx = np.ogrid[-5:6, -5:6]
    foot = ((yy * yy + xx * xx) <= 26)[None]
    for k in range(6):
        image = (field[:, k] + shift) * scale
        local = crop(image)
        lo = ndi.minimum_filter(image, footprint=foot)
        hi = ndi.maximum_filter(image, footprint=foot)
        mid = (np.clip(lo, 0, 255) + np.clip(hi, 0, 255)) * 0.5
        margin = mid - image
        span = np.clip(hi, 0, 255) - np.clip(lo, 0, 255)
        allow = ((crop(margin) > -11) & (crop(span) > ct * 0.25)) | (crop(mid) < 149)
        keep = allow if keep is None else keep | allow
        group.extend([local, crop(mid), crop(span), crop(margin)])
        for arr in [image, margin]:
            # Boundary slope/curvature and directional anisotropy, without absolute coordinates.
            smooth = ndi.gaussian_filter(arr, (0, 0.8, 0.8), truncate=3)
            gy, gx = np.gradient(smooth, axis=(1, 2))
            yy2 = np.gradient(gy, axis=1)
            xx2 = np.gradient(gx, axis=2)
            xy = np.gradient(gy, axis=2)
            group.extend(
                [
                    crop(smooth - arr),
                    crop(np.hypot(gy, gx)),
                    crop(yy2 + xx2),
                    crop(np.hypot(yy2 - xx2, 2 * xy)),
                ]
            )
        if k in (1, 4):
            avg = ndi.uniform_filter(image, (1, 5, 5))
            var = np.maximum(0, ndi.uniform_filter(image**2, (1, 5, 5)) - avg**2)
            group.extend([crop(avg) - local, crop(np.sqrt(var))])
    return np.stack([a[keep] for a in group], axis=1).astype("f4"), keep


def train(blocks):
    # Stratified negative sampling within each block, with inverse sampling weights.
    # Avoid an artificial foreground prior and a large rare-class cutoff shift.
    rng = np.random.default_rng(442)
    xtrain = []
    ytrain = []
    weights = []
    for x, m, truth in blocks:
        y = truth[m]
        a = np.flatnonzero(y)
        b = np.flatnonzero(~y)
        sa = rng.choice(a, min(55000, len(a)), False)
        sb = rng.choice(b, min(85000, len(b)), False)
        idx = np.r_[sa, sb]
        xtrain.append(x[idx])
        ytrain.append(y[idx])
        weights.append(
            np.r_[
                np.full(len(sa), len(a) / max(len(sa), 1)),
                np.full(len(sb), len(b) / max(len(sb), 1)),
            ]
        )
    model = HistGradientBoostingClassifier(
        max_iter=175,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1,
        early_stopping=False,
        random_state=442,
    )
    with threadpool_limits(4):
        model.fit(
            np.concatenate(xtrain), np.concatenate(ytrain), sample_weight=np.concatenate(weights)
        )
    return model


def train_threshold(blocks):
    # Fit the deployed model once. Do not refit it after selecting the cutoff.
    # In each direction fit 12 slices, skip a 4-slice guard, then calibrate on
    # the final 8. Combine the two threshold-centred log-odds for deployment.
    members = []
    for seq in (blocks, blocks[::-1]):
        model = train(seq[:3])
        probs = []
        ys = []
        for x, m, y in seq[4:]:
            probs.append(probability(model, x, m))
            ys.append(y)
        p = np.concatenate(probs)
        y = np.concatenate(ys)
        v = np.arange(20, 81) / 100
        pos = np.sort(p[y])
        neg = np.sort(p[~y])
        tp = len(pos) - np.searchsorted(pos, v)
        fp = len(neg) - np.searchsorted(neg, v)
        loss = 1 - 2 * tp / (tp + fp + len(pos)) + 0.25 * np.abs(tp + fp - len(pos)) / len(pos)
        members.append((model, float(v[np.argmin(loss)])))
    return members, 0.5


def ensemble(members, x, m):
    p = np.zeros(m.shape, "f4")
    for model, t in members:
        q = probability(model, x, m)[m]
        q = np.clip(q, 1e-6, 1 - 1e-6)
        p[m] += np.log(q / (1 - q)) - np.log(t / (1 - t))
    p[m] = 1 / (1 + np.exp(-p[m] / len(members)))
    return p
