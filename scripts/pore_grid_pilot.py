"""Block-grid diagnostic on two disjoint parts of each development calibration stack only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
from denoise import Denoise
from local_model import SETTINGS, XY_HALO, local_fields, median_stack, segment
from validate import ORIGIN, score, stack, verified


def grid(patches, stride, py, px, origin_yx):
    """Average estimates only from patch centres on the chosen source-coordinate lattice."""
    if not 1 <= stride <= 5 or not (0 <= py < stride and 0 <= px < stride):
        raise ValueError("invalid lattice")
    _, _, h, w = patches.shape
    acc = np.zeros_like(patches[:, 0])
    wt = np.zeros((h, w), "u1")
    oy, ox = origin_yx
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cy = (py - oy) % stride
            cx = (px - ox) % stride
            sy = cy + max(0, -(cy + dy) + stride - 1) // stride * stride
            sx = cx + max(0, -(cx + dx) + stride - 1) // stride * stride
            ys = slice(sy, min(h, h - dy), stride)
            xs = slice(sx, min(w, w - dx), stride)
            yd = slice(sy + dy, min(h, h + dy), stride)
            xd = slice(sx + dx, min(w, w + dx), stride)
            acc[:, yd, xd] += patches[:, 12 + dy * 5 + dx, ys, xs]
            wt[yd, xd] += 1
    return acc / np.maximum(wt, 1)


def run(i, cache, gpu):
    raw, _ = verified(cache, i, "raw", False)
    seg, _ = verified(cache, i, "segmented", False)
    spans = [(160, 164), (180, 184)]
    med = np.concatenate([median_stack(raw, i, a, b) for a, b in spans])
    truth = np.concatenate([stack(seg, i, "segmented", a, b) for a, b in spans])
    out = []
    for band in (20, 30, 40):
        p = gpu(med, band / SETTINGS[i][0])
        for stride in (1, 2, 3, 4, 5):
            for py in range(stride):
                for px in range(stride):
                    den = grid(p, stride, py, px, (ORIGIN[1] - XY_HALO, ORIGIN[0] - XY_HALO))
                    pred = segment(local_fields(den, i), SETTINGS[i][2])
                    out.append(
                        dict(
                            band=band,
                            stride=stride,
                            py=py,
                            px=px,
                            fit=score(pred[:4], truth[:4])["dice"],
                            check=score(pred[4:], truth[4:])["dice"],
                        )
                    )
    base = max((v for v in out if v["stride"] == 1), key=lambda v: v["fit"])
    best = max((v for v in out if v["stride"] > 1), key=lambda v: v["fit"])
    print(i, "dense", base, "lattice", best, flush=True)
    return dict(specimen=i, base=base, selected=best, rows=out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with Denoise(ROOT / "build/pores/patches.ptx", patch=True) as gpu:
        for i in (2, 3, 5):
            (a.out / f"sample{i}.json").write_text(
                json.dumps(run(i, a.cache, gpu), indent=2) + "\n", encoding="utf8"
            )
