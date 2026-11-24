"""Geometry × bandwidth predeclared screen of source-window interpretations."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from pore_bernsen_provenance import hashfile
from pore_patch_stats import (
    DEV,
    ROOT,
    SETTINGS,
    Denoise,
    decision,
    median_stack,
    score,
    stack,
    verified,
)
from pore_slice_noise import md

BANDS = (15, 20, 25, 30, 35, 40)
GEOM = ((10, 2), (21, 2), (10, 5), (21, 5))
H = 32
SEAL = ROOT / "pores/evidence/window-units/predictions.json"
PTX = ROOT / "build/pores/window-units.ptx"


def write(p, doc):
    with p.open("x", encoding="utf8", newline="\n") as f:
        json.dump(doc, f, indent=2, allow_nan=False)
        f.write("\n")


def direct(im, y, x, r, p, unit):
    # Independent individual patches, nearest-edge padding only on the source image.
    a = np.pad(im.astype("f8"), r + p, mode="edge")
    base = a[y + r : y + r + 2 * p + 1, x + r : x + r + 2 * p + 1]
    weights = []
    center = []
    for j in range(2 * r + 1):
        for k in range(2 * r + 1):
            patch = a[y + j : y + j + 2 * p + 1, x + k : x + k + 2 * p + 1]
            weights.append(np.mean((patch - base) ** 2))
            center.append(patch[p, p])
    w = np.exp(-np.asarray(weights)[:, None] / (np.array(BANDS, dtype="f8") * unit) ** 2)
    return (w.T @ np.array(center)) / w.sum(0)


def check_numeric(gpu):
    rng = np.random.default_rng(9621)
    im = rng.integers(25000, 42001, (1, 23, 29), dtype="u2")
    rec = []
    for r, p in GEOM:
        got = gpu(im, 160, weighted=int(r == 21), noise=p)[0]
        for y, x in ((0, 0), (12, 14), (20, 27)):
            expected = direct(im[0], y, x, r, p, 160)
            v = got[:, y, x]
            delta = float(np.max(abs(expected - v)))
            if delta > 0.008:
                raise ValueError(("CPU/CUDA geometry or arithmetic mismatch", r, p, y, x, delta))
            rec.append(
                dict(
                    r=r,
                    p=p,
                    y=y,
                    x=x,
                    cpu=expected.tolist(),
                    cuda=v.tolist(),
                    max_abs_native_gray=delta,
                )
            )
    return rec


def frac(row):
    return Fraction(2 * row["tp"], 2 * row["tp"] + row["fp"] + row["fn"])


def gname(r, p):
    return f"{r}-{p}"


def region(raw, i, fit):
    z, x, n = (160, 0, 4) if fit else (176, 152, 8)
    return median_stack(raw, i, z, z + n, H)[:, :, x : x + 104 + 2 * H]


def masks(m, gpu, i, g):
    r, p = g
    den = gpu(m, 1 / SETTINGS[i][0], weighted=int(r == 21), noise=p)
    return [decision(den[:, j], i, 0)[:, H:-H, H:-H] for j in range(len(BANDS))]


def predict(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    records = {}
    prior = json.loads((ROOT / "pores/evidence/bernsen-decisions.json").read_text())["specimens"]
    raws = {}
    with Denoise(PTX, adaptive=True) as gpu:
        audit = check_numeric(gpu)
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            raws[i] = raw
            m = region(raw, i, True)
            t = stack(seg, i, "segmented", 160, 164)[:, :, :104]
            fits = {}
            chosen = {}
            for g in GEOM:
                name = gname(*g)
                vals = masks(m, gpu, i, g)
                fits[name] = {b: score(v, t) for b, v in zip(BANDS, vals, strict=True)}
                chosen[name] = max(BANDS, key=lambda b: frac(fits[name][b]))
            records[i] = dict(fit=fits, band=chosen, raw=rs["sha256"], segmented=ss["sha256"])
            print("fit", i, chosen, flush=True)
        base = gname(*GEOM[0])
        best = max(
            GEOM[1:],
            key=lambda g: sum(
                frac(records[i]["fit"][gname(*g)][records[i]["band"][gname(*g)]]) for i in DEV
            ),
        )
        alt = gname(*best)
        for i in DEV:
            m = region(raws[i], i, False)
            packed = {}
            rec = records[i]
            for arm, g in [("control", GEOM[0]), ("wide", best)]:
                vals = masks(m, gpu, i, g)
                b = rec["band"][gname(*g)]
                packed[arm] = vals[BANDS.index(b)]
                if arm == "control":
                    rec["baseline_h30"] = md(vals[BANDS.index(30)])
                    if rec["baseline_h30"] != prior[str(i)]["masks"]["strict"]:
                        raise ValueError(("Different fixed h30 decisions", i))
            f = out / f"sample{i}.npz"
            np.savez_compressed(f, **packed)
            rec.update(file=hashfile(f), masks={k: md(v) for k, v in packed.items()})
            print("sealed", i, best, flush=True)
    write(
        out / "predictions.json",
        dict(ptx=hashfile(PTX), cpu_gpu=audit, control=base, alternative=alt, specimens=records),
    )


def evaluate(cache, pred, out):
    locked = json.loads(SEAL.read_text())
    records = locked["specimens"]
    if locked != json.loads((pred / "predictions.json").read_text()) or set(records) != set(
        "235"
    ):
        raise ValueError("Wrong seal")
    for k, row in records.items():
        p = pred / f"sample{k}.npz"
        if hashfile(p) != row["file"]:
            raise ValueError("Changed input")
        with np.load(p, allow_pickle=False) as v:
            if {a: md(v[a]) for a in v.files} != row["masks"]:
                raise ValueError("Wrong masks")
    results = {}
    for i in DEV:
        seg, ss = verified(cache, i, "segmented", False)
        if ss["sha256"] != records[str(i)]["segmented"]:
            raise ValueError("Different check source")
        t = stack(seg, i, "segmented", 176, 184)[:, :, 152:]
        with np.load(pred / f"sample{i}.npz", allow_pickle=False) as f:
            a, b = f["control"], f["wide"]
        x, y = score(a, t), score(b, t)
        delta = {k: y[k] - x[k] for k in ("dice", "precision", "recall")}
        hist = np.bincount((4 * t.astype("u1") + 2 * a + b).ravel(), minlength=8)
        results[i] = dict(
            control=x,
            wide=y,
            delta=delta,
            truth_control_wide_histogram=hist.tolist(),
            advance=delta["dice"] >= 0.005 and min(delta["precision"], delta["recall"]) >= -0.005,
        )
        print(i, delta, flush=True)
    write(out, results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("predict", "score"))
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    predict(a.cache, a.out) if a.phase == "predict" else evaluate(a.cache, a.pred, a.out)
