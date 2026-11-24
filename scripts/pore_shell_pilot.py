"""Cross-fitted, calibration-only test of the preregistered one-voxel probability shells."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
import rank_model  # noqa: E402
from denoise import Denoise  # noqa: E402
from local_model import SETTINGS, median_stack  # noqa: E402
from neck_model import HALO, Z_HALO  # noqa: E402
from residual_model import probability  # noqa: E402
from validate import stack, verified  # noqa: E402

FOLDS = {"L": (0, 96), "R": (160, 256)}
ARMS = ("base", "expand", "contract")
FACE = ndi.generate_binary_structure(3, 1)


def components(a):
    labels, n = ndi.label(a, FACE)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    cut = np.unique(
        np.concatenate([np.take(labels, j, axis=k).ravel() for k in range(3) for j in (0, -1)])
    )
    whole = np.ones(n + 1, bool)
    whole[0] = False
    whole[cut] = False
    return labels, sizes, whole


def shells(p):
    base = p >= 0.5
    lab, size, whole = components(base)
    seed = (whole & (size >= 32) & (size <= 160))[lab]
    expand = base | ((~base) & ndi.binary_dilation(seed, structure=FACE) & (p >= 0.35))
    contract = base & ~(seed & ~ndi.binary_erosion(seed, structure=FACE) & (p < 0.65))
    return dict(zip(ARMS, (base, expand, contract), strict=True))


def compare(pred, truth, fold):
    lp, vp, wp = components(pred)
    lr, vr, wr = components(truth)
    ep = wp & (vp >= 64)
    er = wr & (vr >= 64)
    active = ep[lp] & er[lr]
    ids, ns = np.unique(lp[active].astype("i8") * len(vr) + lr[active], return_counts=True)
    edges = []
    for key, h in zip(ids, ns, strict=True):
        p, r = divmod(int(key), len(vr))
        h = int(h)
        union = int(vp[p] + vr[r]) - h
        if 2 * h >= union:
            edges.append((h / union, p, r))
    matched = {}
    used = set()
    for iou, p, r in sorted(edges, reverse=True):
        if r not in matched and p not in used:
            matched[r] = (p, iou)
            used.add(p)
    small = {}
    for r in np.flatnonzero(er & (vr < 128)):
        p, iou = matched.get(int(r), (0, None))
        small[f"{fold}:{r}"] = {
            "reference_voxels": int(vr[r]),
            "pred_voxels": int(vp[p]) if p else None,
            "pred_fold_label": p or None,
            "iou": iou,
        }
    return {
        "eligible_prediction": int(ep.sum()),
        "eligible_reference": int(er.sum()),
        "matches": len(matched),
        "clipped_prediction_ge64": int((~wp[1:] & (vp[1:] >= 64)).sum()),
        "clipped_reference_ge64": int((~wr[1:] & (vr[1:] >= 64)).sum()),
        "small_references": small,
    }


def pack(folds, arm):
    records = [t[arm] for t in folds.values()]
    sums = {
        k: sum(r[k] for r in records)
        for k in ("eligible_reference", "eligible_prediction", "matches")
    }
    sums["missed"] = sums["eligible_reference"] - sums["matches"]
    sums["unmatched"] = sums["eligible_prediction"] - sums["matches"]
    return sums, {k: v for r in records for k, v in r["small_references"].items()}


def summary(folds):
    base, rr = pack(folds, "base")
    out = {"base": base}
    for arm in ARMS[1:]:
        counts, ss = pack(folds, arm)
        if rr.keys() != ss.keys():
            raise ValueError("Different reference population")
        ids = [
            k
            for k, v in rr.items()
            if v["pred_voxels"] is not None and ss[k]["pred_voxels"] is not None
        ]
        p95 = [
            float(
                np.percentile(
                    [
                        100
                        * abs(rows[k]["pred_voxels"] - rows[k]["reference_voxels"])
                        / rows[k]["reference_voxels"]
                        for k in ids
                    ],
                    95,
                )
            )
            if ids
            else None
            for rows in (rr, ss)
        ]
        adequate = len(ids) >= 5
        out[arm] = {
            **counts,
            "small_shared_ids": ids,
            "small_shared_n": len(ids),
            "base_small_shared_p95_pct": p95[0],
            "arm_small_shared_p95_pct": p95[1],
            "decision": "insufficient pairs"
            if not adequate
            else (
                "pass"
                if p95[0] - p95[1] >= 3
                and counts["missed"] <= base["missed"]
                and counts["unmatched"] <= base["unmatched"]
                else "fail"
            ),
        }
    return out


def fold_features(raw_stack, i, side, single, ranked):
    x0, x1 = FOLDS[side]
    med = np.ascontiguousarray(raw_stack[:, :, x0 : x1 + 2 * HALO])
    h = 30 / SETTINGS[i][0]
    fields = np.concatenate([single(med, h), ranked(med, h)], axis=1)
    return [
        rank_model.features(fields[z : z + 4 + 2 * Z_HALO], med[z : z + 4 + 2 * Z_HALO], i)
        for z in range(0, 24, 4)
    ]


def run(cache, i, single, ranked):
    if i not in (2, 3, 5):
        raise ValueError("Development calibration only")
    raw, src = verified(cache, i, "raw", False)
    seg, masksrc = verified(cache, i, "segmented", False)
    med = median_stack(raw, i, 160 - Z_HALO, 184 + Z_HALO, HALO)
    by_fold = {side: fold_features(med, i, side, single, ranked) for side in FOLDS}
    del med
    y = stack(seg, i, "segmented", 160, 184)
    folds = {}
    for fit, check in [("R", "L"), ("L", "R")]:
        start, end = FOLDS[fit]
        truth = y[:, :, start:end].copy()
        model, cut = rank_model.train(
            [(x, m, truth[k * 4 : k * 4 + 4]) for k, (x, m) in enumerate(by_fold[fit])]
        )
        assert cut == 0.5
        p = np.concatenate([probability(model, x, m) for x, m in by_fold[check]])
        pred = shells(p)
        del p
        start, end = FOLDS[check]
        check_truth = y[:, :, start:end]
        folds[check] = {name: compare(vol, check_truth, check) for name, vol in pred.items()}
    return {
        "specimen": i,
        "stage": "calibration only",
        "calibration_z": [160, 184],
        "x_source": {k: [360 + a, 360 + b] for k, (a, b) in FOLDS.items()},
        "raw_sha256": src["sha256"],
        "reference_sha256": masksrc["sha256"],
        "checked_folds": folds,
        "summary": summary(folds),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise SystemExit("Use an empty output directory")
    with (
        Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as single,
        Denoise(ROOT / "build/pores/rankpatch.ptx", ranked=True) as ranked,
    ):
        for i in (2, 3, 5):
            r = run(a.cache, i, single, ranked)
            (a.out / f"sample{i}.json").write_text(
                json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf8"
            )
            print(i, json.dumps(r["summary"]), flush=True)
