"""Calibration-only crossed orientation control, with separated X and Z scoring windows."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
from denoise import Denoise  # noqa: E402
from local_model import SETTINGS, median_stack  # noqa: E402
from validate import score, stack, verified  # noqa: E402


def filter_in_plane(a, axis, op):
    return np.moveaxis(op(np.ascontiguousarray(np.moveaxis(a, axis, 0))), 0, axis)


def decision(a, i, axis):
    scale, shift, ct = SETTINGS[i]
    v = np.clip(np.floor((a + shift) * scale + 0.5), 0, 255).astype("u1")
    yy, xx = np.ogrid[-5:6, -5:6]
    disk = (xx * xx + yy * yy <= 26)[None]
    foot = np.moveaxis(disk, 0, axis)
    lo = ndi.minimum_filter(v, footprint=foot).astype("i2")
    hi = ndi.maximum_filter(v, footprint=foot).astype("i2")
    mid = (lo + hi) // 2
    return np.where(hi - lo < ct, mid < 128, v < mid)


def run(i, cache, gpu):
    raw, _ = verified(cache, i, "raw", False)
    seg, _ = verified(cache, i, "segmented", False)
    rows = {}
    h = 19
    for start, x0, x1, split in [(160, 0, 104, "fit"), (180, 152, 256, "check")]:
        med = median_stack(raw, i, start - h, start + 4 + h, h)[:, :, x0 : x1 + 2 * h]
        truth = stack(seg, i, "segmented", start, start + 4)[:, :, x0:x1]
        sl = (slice(h, -h),) * 3
        for axis in range(3):
            for band in (20, 30, 40):

                def denoise(a, band=band):
                    return gpu(a, band / SETTINGS[i][0])

                result = filter_in_plane(med, axis, denoise)
                for cut in range(3):
                    pred = decision(result, i, cut)[sl]
                    rec = rows.setdefault(
                        (axis, cut, band),
                        dict(
                            nlm=("XY", "XZ", "YZ")[axis], disk=("XY", "XZ", "YZ")[cut], band=band
                        ),
                    )
                    rec[split] = score(pred, truth)["dice"]
    base = max((v for (a, c, _), v in rows.items() if a == c == 0), key=lambda r: r["fit"])
    best = max((v for (a, c, _), v in rows.items() if (a, c) != (0, 0)), key=lambda r: r["fit"])
    print(i, base, best, flush=True)
    return dict(specimen=i, base=base, alternative=best, rows=list(rows.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with Denoise(ROOT / "build/pores/nlm.ptx") as gpu:
        for i in (2, 3, 5):
            (a.out / f"sample{i}.json").write_text(
                json.dumps(run(i, a.cache, gpu), indent=2) + "\n", encoding="utf8"
            )
