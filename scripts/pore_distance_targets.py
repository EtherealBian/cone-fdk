"""Censored radius-four distance to observed opposite digital labels; calibration only."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, digest, stack, verified, write
from pore_census_train import EVID
from scipy import ndimage as ndi


def targets(t):
    if t.ndim != 3 or t.dtype != bool or min(t.shape) < 9:
        raise ValueError("Expected a binary 3D calibration slab with interior margin")
    dist = np.zeros(t.shape, "f4")
    if t.any() and not t.all():
        band = np.zeros_like(t)
        band[4:-4, 4:-4, 4:-4] = True
        for mask, sign in ((t, 1), (~t, -1)):
            d = ndi.distance_transform_edt(mask)
            use = band & mask & (d <= 4)
            dist[use] = sign * (d[use] - 0.5)
    return dist


def describe(d):
    sides = {}
    for k, v in (("pore", d), ("material", -d)):
        extended = (v > 0.5).sum(axis=(1, 2))
        bins = [
            int(np.count_nonzero((v > a) & (v <= b)))
            for a, b in ((0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5))
        ]
        sides[k] = dict(
            centers=sum(bins),
            bins=bins,
            beyond_one=int(extended.sum()),
            beyond_one_by_z=extended.tolist(),
            planes_with_beyond_one=int(np.count_nonzero(extended)),
        )
    return dict(
        sides=sides,
        distance_f32=digest(d),
        sign_f32=digest(np.sign(d) * np.float32(0.5)),
        eligible=all(
            x["centers"] >= 128 and x["beyond_one"] >= 32 and x["planes_with_beyond_one"] >= 4
            for x in sides.values()
        ),
    )


def calibration(cache):
    ref = json.loads((EVID / "fit.json").read_bytes())
    ts, ds, rec = {}, {}, {}
    for i in DEV:
        z, sig = verified(cache, i, "segmented", False)
        t = stack(z, i, "segmented", 160, 184)
        ident = ref["specimens"][str(i)]
        if (
            sig["sha256"] != ident["segmented"]
            or digest(t) != ident["cal_mask"]
            or t.shape != (24, 256, 256)
        ):
            raise ValueError("Wrong calibration mask")
        d = targets(t)
        ts[i], ds[i] = t, d
        rec[str(i)] = dict(cal_mask=digest(t), segmented=sig["sha256"], **describe(d))
    return (
        ts,
        ds,
        dict(
            origin_xyz=[360, 376, 160],
            shape_zyx=[24, 256, 256],
            specimens=rec,
            eligible=all(a["eligible"] for a in rec.values()),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    write(a.out, calibration(a.cache)[2])
