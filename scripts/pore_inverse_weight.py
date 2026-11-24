"""Calibrate a monotone nonlocal weight curve, using native unnormalized statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
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
from threadpoolctl import threadpool_limits


class KernelStats(Denoise):
    def __init__(self, ptx):
        super().__init__(ptx)
        self.channels = 14

    def __call__(self, a, raw_per_unit):
        if a.dtype != np.uint16:
            raise ValueError("Requires exact native uint16 intensity")
        raw = super().__call__(a, raw_per_unit)
        if not np.isfinite(raw).all() or (raw[:, 7:] < 1).any():
            raise ValueError("Invalid patch statistics")
        return raw


def blend(stat, coef):
    coef = np.asarray(coef, dtype="f8")
    if (
        stat.ndim != 4
        or stat.shape[1] != 14
        or coef.shape != (7,)
        or not np.isfinite(coef).all()
        or np.min(coef) < 0
        or not np.isclose(coef.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Invalid mixture")
    # Apply the same coefficients to the unnormalized gray sums and neighbor counts.
    a, b = [np.einsum("b,zbxy->zxy", coef, arr, dtype="f8") for arr in (stat[:, :7], stat[:, 7:])]
    if not (b > 0).all():
        raise ValueError("Empty neighborhood")
    return a / b


def predict(stat, coef, i):
    return decision(blend(stat, coef), i, 0)[:, H:-H, H:-H]


def search(eval_mask):
    basis = np.eye(7)
    current = basis[0]
    initial = eval_mask(current)
    best = initial
    trace = []
    for rnd in range(2):
        start = current.copy()
        for b in range(1, 7):
            for alpha in (0.25, 0.5, 0.75, 1.0):
                coef = (1 - alpha) * start + alpha * basis[b]
                m = eval_mask(coef)
                trace.append(dict(round=rnd, basis=b, alpha=alpha, coef=coef.tolist(), metrics=m))
                # Compare Dice as exact integer fractions, including deterministic ties.
                x, y = (2 * q["tp"] + q["fp"] + q["fn"] for q in (m, best))
                if x == 0 or y == 0:
                    raise ValueError("Empty calibration populations")
                if m["tp"] * y > best["tp"] * x:
                    current, best = coef, m
    return current, initial, best, trace


def run(i, cache, gpu):
    if i not in DEV:
        raise ValueError("Development only")
    raw, rs = verified(cache, i, "raw", False)
    seg, ss = verified(cache, i, "segmented", False)

    def window(a, x):
        med = median_stack(raw, i, a, a + 8, H)[:, :, x : x + 104 + 2 * H]
        truth = stack(seg, i, "segmented", a, a + 8)[:, :, x : x + 104]
        return gpu(med, 1 / SETTINGS[i][0]), truth

    stat, truth = window(160, 0)
    coef, init, best, trace = search(lambda c: score(predict(stat, c, i), truth))
    s, t = window(176, 152)
    b = predict(s, np.eye(7)[0], i)
    a = predict(s, coef, i)
    base, arm = [score(m, t) for m in (b, a)]
    delta = {k: arm[k] - base[k] for k in ("dice", "precision", "recall")}
    transitions = {
        k: int(v.sum())
        for k, v in {
            "fn_fixed": ~b & a & t,
            "fn_added": b & ~a & t,
            "fp_fixed": b & ~a & ~t,
            "fp_added": ~b & a & ~t,
        }.items()
    }
    return dict(
        specimen=i,
        raw=rs["sha256"],
        segmented=ss["sha256"],
        fit_initial=init,
        fit_selected=best,
        fit_trace=trace,
        coefficients=coef.tolist(),
        check_base=base,
        check_arm=arm,
        delta=delta,
        transitions=transitions,
        advance=delta["dice"] >= 0.005 and min(delta["precision"], delta["recall"]) >= -0.005,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = {}
    with KernelStats(ROOT / "build/pores/inverse-weight.ptx") as gpu, threadpool_limits(1):
        for i in DEV:
            r[i] = run(i, a.cache, gpu)
            print(i, r[i]["coefficients"], r[i]["delta"], r[i]["advance"], flush=True)
    with a.out.open("x", encoding="utf8") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
