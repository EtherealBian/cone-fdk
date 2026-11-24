"""Paired preregistered study of patch-statistics preselection; calibration only."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_plane_pilot import (
    ROOT,
    SETTINGS,
    Denoise,
    decision,
    median_stack,
    score,
    stack,
    verified,
)

DEV = (2, 3, 5)
BANDS = (20, 30, 40)
H = 18


def run(i, cache, gpu):
    if i not in DEV:
        raise ValueError("Not a development specimen")
    raw, rs = verified(cache, i, "raw", False)
    seg, ss = verified(cache, i, "segmented", False)
    rec = dict(specimen=i, raw=rs["sha256"], segmented=ss["sha256"], fit={}, check={})
    selected = {}
    for split, a, x in [("fit", 160, 0), ("check", 176, 152)]:
        med = median_stack(raw, i, a, a + 8, H)[:, :, x : x + 104 + 2 * H]
        truth = stack(seg, i, "segmented", a, a + 8)[:, :, x : x + 104]
        arm_masks = []
        for flag, arm in enumerate(("dense", "selected")):
            results = {}
            for band in BANDS if split == "fit" else [selected[arm]]:
                sm = gpu(med, band / SETTINGS[i][0], weighted=flag)
                mask = decision(sm, i, 0)[:, H:-H, H:-H]
                results[band] = score(mask, truth)
                if split == "check":
                    arm_masks.append(mask)
            if split == "fit":
                selected[arm] = max(BANDS, key=lambda k: results[k]["dice"])
            rec[split][arm] = results
        if split == "check":
            old, new = arm_masks
            rec["transitions"] = {
                n: int(np.count_nonzero(q))
                for n, q in {
                    "fn_fixed": truth & ~old & new,
                    "fn_added": truth & old & ~new,
                    "fp_fixed": ~truth & old & ~new,
                    "fp_added": ~truth & ~old & new,
                }.items()
            }
    rec["band"] = selected
    p, q = [rec["check"][k][selected[k]] for k in ("dense", "selected")]
    rec["delta"] = {k: q[k] - p[k] for k in ("dice", "precision", "recall")}
    d = rec["delta"]
    rec["advance"] = bool(
        d["dice"] >= 0.005 and d["precision"] >= -0.005 and d["recall"] >= -0.005
    )
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with (
        a.out.open("x", encoding="utf8") as fp,
        Denoise(ROOT / "build/pores/patch-stats.ptx") as gpu,
    ):
        records = {}
        for i in DEV:
            records[str(i)] = r = run(i, a.cache, gpu)
            print(i, r["band"], r["delta"], r["advance"], flush=True)
        json.dump(records, fp, indent=2, allow_nan=False)
        fp.write("\n")
