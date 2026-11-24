"""Candidate D: combine a local threshold margin with 3-D evidence at narrow necks."""

import numpy as np
from local_model import SETTINGS
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

HALO = 24
Z_HALO = 2


def features(field, median, specimen, halo=HALO):
    """Fields include two real adjacent planes on either side; no Z batch seams."""
    scale, shift, ct = SETTINGS[specimen]
    yy, xx = np.ogrid[-5:6, -5:6]
    disk = (xx * xx + yy * yy <= 26)[None]
    sl = (slice(Z_HALO, -Z_HALO), slice(halo, -halo), slice(halo, -halo))
    group = []
    keep = None
    proxy = None
    for j in range(6):
        img = (field[:, j] + shift) * scale
        val = np.clip(np.floor(img + 0.5), 0, 255)
        low = ndi.minimum_filter(val, footprint=disk)
        high = ndi.maximum_filter(val, footprint=disk)
        mid = np.floor((low + high) * 0.5)
        span = high - low
        margin = mid - img
        pri = np.where(span < ct, 128 - mid, margin)
        viable = ((margin > -14) & (span > ct * 0.2)) | (mid < 153)
        keep = viable[sl].copy() if keep is None else keep | viable[sl]
        for a in (img, low, high, mid - img, span - ct, pri):
            group.append(a[sl])
        if j in (0, 1, 2, 3):
            ave = ndi.uniform_filter(margin, size=(1, 3, 3))
            group.append((ave - margin)[sl])
            # A narrow bridge can have two occupied opposing faces, but little
            # support from the other directions. Preserve that distinction.
            for dist in (1, 2):
                sides = [
                    np.minimum(np.roll(pri, dist, axis=a), np.roll(pri, -dist, axis=a))[sl]
                    for a in (0, 1, 2)
                ]
                group.extend(
                    [
                        sides[0],
                        np.maximum(sides[1], sides[2]),
                        np.minimum(sides[1], sides[2]),
                        np.max(sides, axis=0) - np.median(sides, axis=0),
                    ]
                )
            group.extend(
                [
                    ndi.uniform_filter((pri > 0).astype("f4"), size=w)[sl]
                    for w in ((3, 3, 3), (5, 7, 7), (1, 11, 11))
                ]
            )
            # A proxy defines sampling weights from the *input*, never from a test label.
            if j == 1:
                proxy = group[-2]
    group.append((median[sl].astype("f4") + shift) * scale)
    return np.stack([a[keep] for a in group] + [proxy[keep]], axis=1).astype("f4"), keep


def train(blocks):
    rng = np.random.default_rng(614)
    xs = []
    ys = []
    weights = []
    for x, mask, truth in blocks:
        y = truth[mask]
        ids = []
        ws = []
        for flag, cap in [(True, 55000), (False, 55000)]:
            entries = np.flatnonzero(y == flag)
            chosen = rng.choice(entries, min(cap, len(entries)), False)
            ids.append(chosen)
            ws.append(np.full(len(chosen), len(entries) / max(1, len(chosen))))
        idx = np.concatenate(ids)
        sampled = x[idx]
        xs.append(sampled)
        ys.append(y[idx])
        # Large pore interiors must not overwhelm small-pore boundary evidence.
        weights.append(np.concatenate(ws) * np.where(sampled[:, -1] < 0.35, 3.0, 1.0))
    model = HistGradientBoostingClassifier(
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=24,
        l2_regularization=1,
        early_stopping=False,
        random_state=614,
    )
    with threadpool_limits(4):
        model.fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(weights))
    return model, 0.5
