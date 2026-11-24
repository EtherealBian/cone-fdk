"""Identify median footprint candidates against a localized grayscale intermediate."""

import argparse
import io
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_axial_raw import ROOT, digest, sha, verified, write
from pore_figure_audit import render
from pore_figure_gray import RANGES, REG, limits, metric
from pore_figure_gray import E as GRAY
from scipy import ndimage as ndi

E = ROOT / "pores/evidence/median-source"
AX = {"X": 2, "Y": 1, "Z": 0}
ORDERS = ["".join(x) for x in itertools.permutations("XYZ")]
ALTS = ["faces7", "edges19", "xy9", *ORDERS]


def footprint(k):
    z, y, x = np.mgrid[-1:2, -1:2, -1:2]
    dist = x * x + y * y + z * z
    if k == "cube27":
        return dist <= 3
    if k == "faces7":
        return dist <= 1
    if k == "edges19":
        return dist <= 2
    if k == "xy9":
        return z == 0
    raise ValueError("Unknown footprint")


def native(raw, k):
    if k in ORDERS:
        out = raw
        for name in k:
            size = [1, 1, 1]
            size[AX[name]] = 3
            out = ndi.median_filter(out, size=size, mode="nearest")
        return out[1]
    return ndi.median_filter(raw, footprint=footprint(k), mode="nearest")[1]


def scalar(c, k):
    if k in ORDERS:
        for name in k:
            c = np.sort(c, axis=AX[name]).take([1], axis=AX[name])
        return int(c.item())
    v = np.sort(c[footprint(k)])
    return int(v[len(v) // 2])


def predict(cache, out):
    d = json.loads((GRAY / "predictions.json").read_text())
    box = limits(d["xy0_scale"], (112, 428))
    path, r = verified(cache, 2, "raw", False)
    z = d["z"]
    if d["raw"] != r["sha256"] or z != 447 or box[0] < 616:
        raise ValueError("Different source")
    raw = []
    with ZipFile(path) as archive:
        for j in range(z - 1, z + 2):
            with Image.open(
                io.BytesIO(archive.read(f"set1sample2raw/set1sample2raw_{j:04d}.tif"))
            ) as im:
                raw.append(np.array(im.crop(box)))
    raw = np.stack(raw)
    cut = np.s_[24:-24, 24:-24]
    a = {"original": raw[1][cut].copy()}
    rng = np.random.default_rng(9529)
    ny, nx = a["original"].shape
    coords = (
        np.column_stack(np.unravel_index(rng.choice(ny * nx, 256, replace=False), (ny, nx))) + 24
    )
    for k in ["cube27", *ALTS]:
        plane = native(raw, k)
        for y, x in coords:
            if scalar(raw[:, y - 1 : y + 2, x - 1 : x + 2], k) != plane[y, x]:
                raise ValueError(("Independent local rank disagrees", k, y, x))
        a[k] = plane[cut].copy()
    if (
        digest(raw) != d["raw_crop"]
        or digest(a["cube27"]) != d["fields"]["median"]
        or digest(a["original"]) != d["fields"]["original"]
    ):
        raise ValueError("Different native gray planes")
    out.mkdir(parents=True, exist_ok=False)
    f = out / "fields.npz"
    np.savez_compressed(f, **a)
    write(
        out / "predictions.json",
        dict(
            specimen=2,
            z=z,
            file=sha(f),
            gray_seal=sha(GRAY / "predictions.json"),
            raw_crop=digest(raw),
            scalar_positions=coords.tolist(),
            verified_per_arm=len(coords),
            fields={k: digest(v) for k, v in a.items()},
        ),
    )


def evaluate(pred, panels):
    seal = json.loads((E / "predictions.json").read_text())
    d = json.loads((GRAY / "predictions.json").read_text())
    r = json.loads(REG.read_text())
    if (
        seal != json.loads((pred / "predictions.json").read_text())
        or sha(pred / "fields.npz") != seal["file"]
        or sha(GRAY / "predictions.json") != seal["gray_seal"]
        or sha(REG) != d["registration"]
    ):
        raise ValueError("Unknown seal")
    with np.load(pred / "fields.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if {k: digest(v) for k, v in a.items()} != seal["fields"]:
        raise ValueError("Different fields")
    p = {}
    for k, h in r["panels"].items():
        if sha(panels / f"{k}.png") != h:
            raise ValueError("Different paper crop")
        with Image.open(panels / f"{k}.png") as im:
            p[k] = np.array(im, dtype="f8")
    res = {}
    for split, (lo, hi) in RANGES.items():
        yy, xx = np.mgrid[80:424, lo:hi]
        raw, med = [p[k][yy, xx] for k in ("raw", "median")]
        diff = med - raw
        g = np.hypot(*np.gradient(med))
        cutoff = np.median(g)
        masks = {"all": np.ones_like(med, bool), "weak": g <= cutoff, "strong": g > cutoff}
        planes = {k: render(v, yy, xx, d) for k, v in a.items()}
        res[split] = {
            k: {grp: metric(v, planes["original"], med, diff, m) for grp, m in masks.items()}
            for k, v in planes.items()
        }
    control = json.loads((ROOT / "pores/evidence/patch-profile/check.json").read_text())["paper"][
        "parts"
    ]["check"]["median_control"]["native_median"]["all"]["change_rmse"]
    right = res["check"]
    orig = right["cube27"]["all"]
    best = min(ALTS, key=lambda k: res["fit"][k]["all"]["change_rmse"])
    chosen = right[best]["all"]
    ratio = chosen["change_rmse"] / orig["change_rmse"]
    if abs(orig["change_rmse"] - control) > 1e-12:
        raise ValueError("Different median-change control")
    return dict(
        partitions=res,
        selected=best,
        check_ratio=ratio,
        advance=ratio <= 0.9 and chosen["absolute_rmse"] <= orig["absolute_rmse"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("predict", "score"))
    for k in ("cache", "pred", "panels", "out"):
        p.add_argument("--" + k, required=k == "out", type=Path)
    a = p.parse_args()
    if a.phase == "predict":
        predict(a.cache, a.out)
    else:
        d = evaluate(a.pred, a.panels)
        write(a.out, d)
        print(d["selected"], d["check_ratio"], d["advance"])
