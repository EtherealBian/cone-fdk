"""Isolate search support and bandwidth using only separated calibration windows."""

import argparse
import json
from pathlib import Path

from pore_patch_stats import (
    DEV,
    ROOT,
    SETTINGS,
    Denoise,
    H,
    decision,
    median_stack,
    score,
    stack,
    verified,
)


def run(i, cache, gpu):
    if i not in DEV:
        raise ValueError("Development only")
    raw, rs = verified(cache, i, "raw", False)
    seg, ss = verified(cache, i, "segmented", False)
    record = dict(
        specimen=i, raw=rs["sha256"], segmented=ss["sha256"], fit={}, check={}, selected={}
    )
    bands = range(20, 41)
    for split, a, x in [("fit", 160, 0), ("check", 176, 152)]:
        med = median_stack(raw, i, a, a + 8, H)[:, :, x : x + 104 + 2 * H]
        truth = stack(seg, i, "segmented", a, a + 8)[:, :, x : x + 104]
        for disk, name in enumerate(("square", "disk")):
            rscores = {}
            for band in (
                bands
                if split == "fit"
                else sorted({record["selected"][name]} | ({30} if not disk else set()))
            ):
                image = gpu(med, band / SETTINGS[i][0], weighted=disk)
                mask = decision(image, i, 0)[:, H:-H, H:-H]
                rscores[band] = score(mask, truth)
            if split == "fit":
                record["selected"][name] = max(bands, key=lambda b: rscores[b]["dice"])
            record[split][name] = rscores
    sf, df = record["selected"].values()
    c = record["check"]
    out = {}
    for name, p, q in [
        ("fine_band", c["square"][30], c["square"][sf]),
        ("disk", c["square"][sf], c["disk"][df]),
    ]:
        delta = {k: q[k] - p[k] for k in ("dice", "precision", "recall")}
        out[name] = dict(
            delta=delta,
            advance=delta["dice"] >= 0.005 and min(delta["precision"], delta["recall"]) >= -0.005,
        )
    record["comparisons"] = out
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rec = {}
    with Denoise(ROOT / "build/pores/search-support.ptx") as gpu:
        for i in DEV:
            r = run(i, a.cache, gpu)
            rec[i] = r
            print(i, r["selected"], r["comparisons"], flush=True)
    with a.out.open("x", encoding="utf8") as f:
        json.dump(rec, f, indent=2, allow_nan=False)
        f.write("\n")
