"""Numerical usability screen for public vendor panels; no CT labels or archives."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def describe(a, b, hist=False):
    d = a.astype("i2") - b.astype("i2")
    res = dict(
        n=d.size,
        mae=float(np.abs(d).mean()),
        exact=float((d == 0).mean()),
        within_one=float((np.abs(d) <= 1).mean()),
    )
    if hist:
        keys, n = np.unique(d, return_counts=True)
        res["signed_error_histogram"] = {int(k): int(v) for k, v in zip(keys, n, strict=True)}
    return res


def controls(a, b, c):
    s = np.s_[12:-12, 12:-12]
    out = {}
    for r in (1, 3):
        out[f"median_radius_{r}"] = describe(ndi.median_filter(a, size=2 * r + 1)[s], c[s])
    for r in (3, 7):
        sm = ndi.uniform_filter(a.astype("f8"), size=2 * r + 1)
        for name, v in [("nearest", np.floor(sm + 0.5)), ("truncate", np.floor(sm + 1e-10))]:
            out[f"box_radius_{r}_{name}"] = describe(v[s], b[s])
    return out, all(
        any(v["within_one"] >= 0.99 for k, v in out.items() if k.startswith(pre))
        for pre in ("median", "box")
    )


def nlm_experiment(src, ref):
    a = src.astype("f8")
    h, w = a.shape
    pad = np.pad(a, 3, mode="edge")
    terms = []
    for j in range(7):
        for i in range(7):
            neighbor = pad[j : j + h, i : i + w]
            d = ndi.uniform_filter((a - neighbor) ** 2, size=7)
            terms.append((neighbor, np.maximum(d, 0)))

    def estimate(bw, sl):
        den = np.zeros_like(a[sl])
        num = den.copy()
        for v, d in terms:
            weight = np.exp(-d[sl] / bw**2)
            num += weight * v[sl]
            den += weight
        return np.floor(num / den + 0.5)

    left = np.s_[12:-12, 12 : w // 3]
    right = np.s_[12:-12, 2 * w // 3 : -12]
    fit = {k: describe(estimate(k, left), ref[left]) for k in range(1, 97)}
    bw = min(fit, key=lambda k: fit[k]["mae"])
    return dict(fit=fit, bandwidth=bw, check=describe(estimate(bw, right), ref[right], True))


def run(folder):
    rec = {"source": "ImageDev 2026-1 Denoising Figure 1", "panels": []}
    planes = []
    for n in range(1, 5):
        p = folder / f"panel{n}.png"
        with Image.open(p) as im:
            a = np.array(im)
            ok = (
                im.mode == "L"
                or (
                    im.mode == "RGB"
                    and np.array_equal(a[..., 0], a[..., 1])
                    and np.array_equal(a[..., 0], a[..., 2])
                )
            ) and "transparency" not in im.info
            rec["panels"].append(
                dict(
                    panel=n,
                    mode=im.mode,
                    size=list(im.size),
                    gray_opaque=ok,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
            )
            planes.append(a if a.ndim == 2 else a[..., 0])
    if not all(
        p["gray_opaque"] and p["size"] == rec["panels"][0]["size"] and min(p["size"]) > 48
        for p in rec["panels"]
    ):
        rec["qualified"] = False
        rec["reason"] = "Incompatible panel geometry, color or transparency"
        return rec
    rec["controls"], rec["qualified"] = controls(*planes[:3])
    if rec["qualified"]:
        rec["nlm"] = nlm_experiment(planes[0], planes[3])
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = run(a.folder)
    with a.out.open("x", encoding="utf8") as f:
        json.dump(r, f, indent=2, allow_nan=False)
        f.write("\n")
    print(r["qualified"], r.get("reason", r.get("controls")), flush=True)
