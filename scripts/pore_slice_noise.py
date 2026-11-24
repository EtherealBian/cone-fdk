"""Measure raw off-ROI slice noise; seal predictions separately from calibration scoring."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
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
from scipy import ndimage as ndi
from validate import SPEC

PROBES = [(x, y) for y in (328, 648) for x in (312, 632)]
SEAL = ROOT / "pores/evidence/slice-noise-predictions.json"


def measures(a):
    if a.shape != (3, 34, 34) or a.dtype != np.uint16:
        raise ValueError("Expected native 3 x 34 x 34 context")
    filtered = ndi.median_filter(a, size=3)[1, 1:-1, 1:-1]
    windows = np.lib.stride_tricks.sliding_window_view(a, (3, 3, 3)).reshape(32, 32, 27)
    direct = np.sort(windows, axis=-1)[:, :, 13]
    if not np.array_equal(direct, filtered):
        raise ValueError("Direct order-statistic replay disagrees")
    r = a[1, 1:-1, 1:-1].astype("i4") - filtered
    return float(np.median(np.abs(r - np.median(r))))


def scales(path, i, a):
    if i not in DEV or a not in (160, 176):
        raise ValueError("Calibration only")
    prefix = f"set1sample{i}raw"
    layers = []
    with ZipFile(path) as z:
        for k in range(a - 1, a + 9):
            with Image.open(io.BytesIO(z.read(f"{prefix}/{prefix}_{k:04d}.tif"))) as im:
                if im.size != SPEC[i][:2]:
                    raise ValueError("Different native image")
                layers.append(
                    [np.asarray(im.crop((x - 1, y - 1, x + 33, y + 33))) for x, y in PROBES]
                )
    mat = np.stack(layers)
    return [[measures(mat[j : j + 3, k]) for k in range(4)] for j in range(8)]


def bands(fit, check):
    a, b = [np.asarray(q, dtype="f8") for q in (fit, check)]
    if any(q.shape != (8, 4) for q in (a, b)):
        raise ValueError("Different probe population")
    f, c = [np.median(q, axis=1) for q in (a, b)]
    if not np.isfinite(np.r_[f, c]).all() or np.min(np.r_[f, c]) <= 0:
        return None
    return 30 * np.clip(c / np.median(f), 0.5, 2)


def md(mask):
    if mask.shape != (8, 256, 104) or mask.dtype != bool:
        raise ValueError("Unexpected check population")
    return hashlib.sha256(np.packbits(mask.ravel(), bitorder="little")).hexdigest()


def predict(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    with Denoise(ROOT / "build/pores/search-support.ptx") as gpu:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            f, c = [scales(raw, i, a) for a in (160, 176)]
            h = bands(f, c)
            r = records[str(i)] = dict(
                raw=rs["sha256"],
                fit_mad=f,
                check_mad=c,
                anchor=float(np.median(np.median(f, axis=1))),
                band=None if h is None else h.tolist(),
                masks={},
            )
            if h is None:
                continue
            m = median_stack(raw, i, 176, 184, H)[:, :, 152 : 256 + 2 * H]
            masks = {}
            for arm, v in [("fixed", np.full(8, 30.0)), ("scaled", h)]:
                image = np.concatenate(
                    [gpu(q[None], k / SETTINGS[i][0]) for q, k in zip(m, v, strict=True)]
                )
                mask = decision(image, i, 0)[:, H:-H, H:-H]
                r["masks"][arm] = md(mask)
                masks[arm] = mask
            np.savez_compressed(out / f"sample{i}.npz", **masks)
    return dict(specimens=records, probe_xy=PROBES, fit_z=[160, 168], check_z=[176, 184])


def evaluate(cache, pred):
    doc = json.loads((pred / "predictions.json").read_text())
    if doc != json.loads(SEAL.read_text()) or set(doc["specimens"]) != {str(i) for i in DEV}:
        raise ValueError("Not the published calibration seal")
    results = {}
    for i in DEV:
        r = doc["specimens"][str(i)]
        if r["band"] is None:
            results[str(i)] = dict(unsupported=True, advance=False)
            continue
        with np.load(pred / f"sample{i}.npz", allow_pickle=False) as z:
            base, arm = [z[k] for k in ("fixed", "scaled")]
        if dict(fixed=md(base), scaled=md(arm)) != r["masks"]:
            raise ValueError("Check masks changed")
        seg, ss = verified(cache, i, "segmented", False)
        truth = stack(seg, i, "segmented", 176, 184)[:, :, 152:]
        b, a = [score(m, truth) for m in (base, arm)]
        delta = {k: a[k] - b[k] for k in ("dice", "precision", "recall")}
        transitions = {
            k: int(q.sum())
            for k, q in dict(
                fn_fixed=~base & arm & truth,
                fn_added=base & ~arm & truth,
                fp_fixed=base & ~arm & ~truth,
                fp_added=~base & arm & ~truth,
            ).items()
        }
        results[str(i)] = dict(
            segmented=ss["sha256"],
            fixed=b,
            scaled=a,
            delta=delta,
            transitions=transitions,
            advance=delta["dice"] >= 0.005 and min(delta["precision"], delta["recall"]) >= -0.005,
        )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["predict", "score"])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.stage == "score" and a.out is None:
        ap.error("score requires --out")
    doc = predict(a.cache, a.pred) if a.stage == "predict" else evaluate(a.cache, a.pred)
    dest = a.pred / "predictions.json" if a.stage == "predict" else a.out
    with dest.open("x", encoding="utf8") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.write("\n")
