"""Candidate E: rank-adaptive image estimates and component-balanced calibration edges."""

import numpy as np
from local_model import SETTINGS
from neck_model import HALO, Z_HALO
from neck_model import features as neck_features
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def features(fields, median, specimen, halo=HALO):
    base, mask = neck_features(fields[:, :6], median, specimen, halo)
    scale, shift, ct = SETTINGS[specimen]
    yy, xx = np.ogrid[-5:6, -5:6]
    disk = (yy * yy + xx * xx <= 26)[None]
    roi = (slice(Z_HALO, -Z_HALO), slice(halo, -halo), slice(halo, -halo))
    extra = []
    for k in range(6, 12):
        img = (fields[:, k] + shift) * scale
        v = np.clip(np.floor(img + 0.5), 0, 255)
        lo = ndi.minimum_filter(v, footprint=disk)
        hi = ndi.maximum_filter(v, footprint=disk)
        mid = np.floor((lo + hi) * 0.5)
        contrast = hi - lo
        margin = mid - img
        gy, gx = np.gradient(img, axis=(1, 2))
        # First-order boundary displacement estimate, measured in pixels (capped).
        projected = np.clip(margin / np.maximum(np.hypot(gy, gx), 0.5), -4, 4)
        support = np.where(contrast < ct, mid < 128, margin > 0)
        dev = ndi.uniform_filter(support.astype("f4"), size=(3, 3, 3))
        for a in (img, margin, contrast - ct, projected, dev):
            extra.append(a[roi][mask])
    return np.column_stack([base, *extra]).astype("f4", copy=False), mask


def edge_weights(truth):
    """Use calibration-only components; give the same multiplier to their nearby background."""
    lab, n = ndi.label(truth)
    result = np.ones(truth.shape, "f4")
    if not n:
        return result
    counts = np.bincount(lab.ravel())
    clipped = np.unique(
        np.concatenate([np.take(lab, i, axis=a).ravel() for a in range(3) for i in (0, -1)])
    )
    per = np.clip(np.sqrt(1024 / np.maximum(counts, 1)), 1, 4).astype("f4")
    per[counts < 16] = 1
    per[clipped] = 1
    per[0] = 1
    dist, ix = ndi.distance_transform_edt(~truth, return_indices=True)
    border = (dist <= 2) & (ndi.distance_transform_edt(truth) <= 2)
    result[border] = per[lab[tuple(ix)][border]]
    return result


def train(blocks):
    truth = np.concatenate([y for _, _, y in blocks])
    balance = edge_weights(truth)
    rng = np.random.default_rng(733)
    xx = []
    yy = []
    ww = []
    z = 0
    for x, mask, y in blocks:
        label = y[mask]
        w = balance[z : z + len(y)][mask]
        z += len(y)
        for flag, cap in ((True, 55000), (False, 85000)):
            ids = np.flatnonzero(label == flag)
            if len(ids):
                take = rng.choice(ids, min(cap, len(ids)), replace=False)
                xx.append(x[take])
                yy.append(label[take])
                ww.append(w[take] * (len(ids) / len(take)))
    model = HistGradientBoostingClassifier(
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=24,
        l2_regularization=1,
        early_stopping=False,
        random_state=733,
    )
    with threadpool_limits(4):
        model.fit(np.concatenate(xx), np.concatenate(yy), sample_weight=np.concatenate(ww))
    return model, 0.5
