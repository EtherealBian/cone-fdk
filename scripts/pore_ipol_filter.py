"""Build and run the cited historical reference denoiser; no reserve input is accepted."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import numpy as np
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

ARCHIVE = "649ad40b8676e29309c478e351bb95ab7f1b2733a4994b94094b29f4758b6902"
SEAL = ROOT / "pores/evidence/ipol-decisions.json"
H = 24
SIGMAS = (10, 15, 20, 25, 30, 35, 40)


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build(archive, compiler, work):
    if sha(archive) != ARCHIVE:
        raise ValueError("Unexpected source archive")
    names = [
        "libdenoising.cpp",
        "libdenoising.h",
        "libauxiliar.cpp",
        "libauxiliar.h",
        "mt19937ar.c",
        "mt19937ar.h",
    ]
    work.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as t:
        for n in names:
            m = t.getmember("nlmeansC/" + n)
            if not m.isfile():
                raise ValueError("Expected upstream source file")
            (work / n).write_bytes(t.extractfile(m).read())
    adapter = ROOT / "scripts/ipol_plane.cpp"
    dll = (work / "denoiser.dll").resolve()
    env = os.environ.copy()
    env["PATH"] = str(compiler.resolve().parent) + os.pathsep + env["PATH"]
    subprocess.run(
        [
            str(compiler.resolve()),
            "-O3",
            "-std=gnu++14",
            "-include",
            "cmath",
            "-ffp-contract=off",
            "-shared",
            "-static",
            "-Wl,--no-insert-timestamp",
            "-I.",
            str(adapter),
            "libdenoising.cpp",
            "libauxiliar.cpp",
            "mt19937ar.c",
            "-o",
            str(dll),
        ],
        cwd=work,
        env=env,
        check=True,
    )
    return dll, dict(
        archive=ARCHIVE,
        adapter=hashlib.sha256(adapter.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        library=sha(dll),
        compiler=subprocess.check_output(
            [str(compiler), "--version"], env=env, text=True
        ).splitlines()[0],
    )


class Filter:
    def __init__(self, dll):
        self.f = ctypes.CDLL(str(dll.resolve())).ipol_plane
        arr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
        self.f.argtypes = [arr, arr, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        self.f.restype = None

    def __call__(self, v, sigma):
        v = np.ascontiguousarray(v, dtype="f4")
        if (
            v.ndim != 2
            or min(v.shape) < 5
            or not np.isfinite(v).all()
            or not np.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError("Invalid input plane or sigma")
        out = np.full_like(v, np.nan)
        self.f(v, out, v.shape[1], v.shape[0], float(sigma))
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite output")
        return out


def interior_pixel(image, y, x, sigma):
    # Gather all 25 patch-centre votes at this one pixel; compute weights in double.
    if min(y, x, image.shape[0] - 1 - y, image.shape[1] - 1 - x) < 14:
        raise ValueError("Expected real support")
    v = np.asarray(image, "f8")
    wins = np.lib.stride_tricks.sliding_window_view(v, (5, 5))
    votes = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cy, cx = y + dy, x + dx
            d = np.mean(
                (wins[cy - 12 : cy + 9, cx - 12 : cx + 9] - wins[cy - 2, cx - 2]) ** 2,
                axis=(2, 3),
            )
            d = np.maximum(0, d - 2 * sigma**2) / (0.6 * sigma) ** 2
            w = np.where(d >= 29, 0, np.exp(-d))
            w[10, 10] = 0
            w[10, 10] = w.max()
            if w.sum() > 1e-8:
                votes.append((w * v[y - 10 : y + 11, x - 10 : x + 11]).sum() / w.sum())
    return float(np.mean(votes)) if votes else float(v[y, x])


def audit(op):
    c = np.full((33, 35), 128, "f4")
    a = op(c, 20)
    if not np.array_equal(a, c):
        raise ValueError("Constant-image invariance failed")
    rng = np.random.default_rng(173)
    v = rng.integers(80, 175, (37, 39)).astype("f4")
    v[12:23, 18:25] += 24
    b = op(v, 18)
    rows = []
    for y, x in [(14, 14), (18, 20), (21, 23)]:
        e = interior_pixel(v, y, x, 18)
        rows.append([y, x, e, float(b[y, x])])
        if abs(e - b[y, x]) > 0.001:
            raise ValueError("Independent scalar audit disagrees")
    return rows


def predict(cache, out, archive, compiler):
    out.mkdir(parents=True, exist_ok=False)
    dll, tools = build(archive, compiler, out / "upstream")
    op = Filter(dll)
    checks = audit(op)
    records = {}
    with Denoise(ROOT / "build/pores/search-support.ptx") as gpu:
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            tm = stack(seg, i, "segmented", 160, 164)[:, :, :104]
            fit = {}
            chosen = None
            for fit_phase, x, a, b in [(True, 0, 160, 164), (False, 152, 176, 184)]:
                m = median_stack(raw, i, a, b, H)[:, :, x : x + 104 + 2 * H]
                sl = (slice(None), slice(H, -H), slice(H, -H))
                masks = {}
                for s in SIGMAS if fit_phase else (chosen,):
                    den = np.stack([op(u, s / SETTINGS[i][0]) for u in m])
                    mask = decision(den, i, 0)[sl]
                    if fit_phase:
                        fit[s] = score(mask, tm)
                    else:
                        masks["ipol"] = mask
                if fit_phase:
                    chosen = max(SIGMAS, key=lambda s: fit[s]["dice"])
                    continue
                masks["fixed"] = decision(gpu(m, 30 / SETTINGS[i][0]), i, 0)[sl]
                np.savez_compressed(out / f"sample{i}.npz", **masks)
                h = {k: md(v) for k, v in masks.items()}
                prior = json.loads((ROOT / "pores/evidence/bernsen-decisions.json").read_text())[
                    "specimens"
                ][str(i)]
                if h["fixed"] != prior["masks"]["strict"]:
                    raise ValueError("Changing source halo changed the fixed baseline")
                records[i] = dict(
                    raw=rs["sha256"], segmented=ss["sha256"], fit=fit, sigma=chosen, masks=h
                )
                print(i, "sealed", chosen, h, flush=True)
    return dict(tools=tools, synthetic_pixel_audit=checks, specimens=records)


def evaluate(cache, out):
    d = json.loads((out / "predictions.json").read_text())
    r = {}
    if d != json.loads(SEAL.read_text()) or set(d["specimens"]) != set("235"):
        raise ValueError("Not the public prediction seal")
    for i in DEV:
        row = d["specimens"][str(i)]
        with np.load(out / f"sample{i}.npz") as z:
            a, b = z["fixed"], z["ipol"]
        if dict(fixed=md(a), ipol=md(b)) != row["masks"]:
            raise ValueError("Masks changed")
        seg, ss = verified(cache, i, "segmented", False)
        if row["segmented"] != ss["sha256"]:
            raise ValueError("Wrong reference")
        t = stack(seg, i, "segmented", 176, 184)[:, :, 152:]
        x, y = score(a, t), score(b, t)
        delta = {k: y[k] - x[k] for k in ("dice", "precision", "recall")}
        r[i] = dict(
            fixed=x,
            ipol=y,
            delta=delta,
            unchanged_fp=int((a & b & ~t).sum()),
            unchanged_fn=int((~a & ~b & t).sum()),
            advance=delta["dice"] >= 0.005 and min(delta["precision"], delta["recall"]) >= -0.005,
        )
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["predict", "score"])
    for k in ("cache", "out", "archive", "compiler"):
        ap.add_argument("--" + k, required=k in ("cache", "out"), type=Path)
    a = ap.parse_args()
    pred = a.phase == "predict"
    if pred and not (a.compiler and a.archive):
        ap.error("Provide the pinned archive and MinGW g++ path")
    d = predict(a.cache, a.out, a.archive, a.compiler) if pred else evaluate(a.cache, a.out)
    with (a.out / ("predictions.json" if pred else "results.json")).open("x") as f:
        json.dump(d, f, indent=2, allow_nan=False)
        f.write("\n")
