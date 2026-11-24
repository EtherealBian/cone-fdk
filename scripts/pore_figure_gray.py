"""Raw-field seal then gray-change comparison on the source-localized Sample 2 illustration."""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_axial_raw import ROOT, SETTINGS, Denoise, digest, sha, verified, write
from pore_figure_register import ncc, samples
from pore_window_units import BANDS, PTX
from scipy import ndimage as ndi

E = ROOT / "pores/evidence/figure-gray"
REG = ROOT / "pores/evidence/figure-locator/registration.json"
RANGES = {"fit": (112, 216), "check": (232, 428)}


def limits(par, xs, halo=24):
    x, y, s = par
    a, b = xs
    # Upper bounds are exclusive: bilinear accesses floor(last)+1.
    return [
        int(np.floor(x + s * a)) - halo,
        int(np.floor(y + s * 80)) - halo,
        int(np.floor(x + s * (b - 1))) + 2 + halo,
        int(np.floor(y + s * 423)) + 2 + halo,
    ]


def predict(cache, out):
    r = json.loads(REG.read_text())
    par = r["source_xy0_scale"]
    z = r["z"]
    raw, rs = verified(cache, 2, "raw", False)
    if not r["advance"] or r["raw"] != rs["sha256"] or z != 447:
        raise ValueError("Wrong registration")
    support = {k: limits(par, v) for k, v in RANGES.items()}
    box = limits(par, (112, 428))
    x, y, x1, y1 = box
    if x < 616 or min(x, y) < 0 or x1 >= 980 or y1 >= 1010:
        raise ValueError("Out of development diagnostic support")
    frames = []
    with ZipFile(raw) as archive:
        for j in range(z - 1, z + 2):
            with Image.open(
                io.BytesIO(archive.read(f"set1sample2raw/set1sample2raw_{j:04d}.tif"))
            ) as im:
                frames.append(np.array(im.crop(box)))
    raw_vol = np.stack(frames)
    med = ndi.median_filter(raw_vol, size=3, mode="nearest")[1]
    cut = np.s_[24:-24, 24:-24]
    fields = {"original": raw_vol[1][cut].copy(), "median": med[cut].copy()}
    with Denoise(PTX, adaptive=True) as gpu:
        for radius in (10, 21):
            vals = gpu(med[None], 1 / SETTINGS[2][0], weighted=int(radius == 21), noise=2)[0]
            for j, h in enumerate(BANDS):
                fields[f"r{radius}h{h}"] = vals[j][cut].copy()
    out.mkdir(parents=True, exist_ok=False)
    f = out / "fields.npz"
    np.savez_compressed(f, **fields)
    write(
        out / "predictions.json",
        dict(
            specimen=2,
            z=z,
            ptx=sha(PTX),
            raw=rs["sha256"],
            raw_crop=digest(raw_vol),
            raw_support_xyxy=support,
            field_origin_xy=[x + 24, y + 24],
            fields={k: digest(v) for k, v in fields.items()},
            file=sha(f),
            registration=sha(REG),
            xy0_scale=par,
            slope=r["intensity_slope"],
            intercept=r["intensity_intercept"],
        ),
    )


def metric(field, base, published, observed, mask):
    diff = (field - base - observed)[mask]
    v = field[mask]
    p = published[mask]
    return dict(
        count=int(len(v)),
        change_rmse=float(np.sqrt(np.mean(diff**2))),
        change_bias=float(diff.mean()),
        absolute_rmse=float(np.sqrt(np.mean((v - p) ** 2))),
        absolute_ncc=ncc(v, p),
    )


def evaluate(pred, panels, out):
    doc = json.loads((E / "predictions.json").read_text())
    reg = json.loads(REG.read_text())
    f = pred / "fields.npz"
    if (
        doc != json.loads((pred / "predictions.json").read_text())
        or sha(f) != doc["file"]
        or sha(REG) != doc["registration"]
        or sha(panels / "median.png") != reg["panels"]["median"]
    ):
        raise ValueError("Changed seal or published median")
    with np.load(f, allow_pickle=False) as v:
        fields = {k: v[k] for k in v.files}
    if {k: digest(v) for k, v in fields.items()} != doc["fields"]:
        raise ValueError("Changed native fields")
    with Image.open(panels / "nlm.png") as im:
        pub = np.asarray(im, dtype="f8")
    with Image.open(panels / "median.png") as im:
        med = np.asarray(im, dtype="f8")
    if pub.shape != (448, 448) or med.shape != pub.shape:
        raise ValueError("Unexpected published dimensions")
    x, y, s = doc["xy0_scale"]
    ox, oy = doc["field_origin_xy"]
    result = {}
    arms = [f"r{r}h{h}" for r in (10, 21) for h in BANDS]
    for part, (l, u) in RANGES.items():
        yy, xx = np.mgrid[80:424, l:u]
        xy = (x - ox, y - oy, s)
        if any(
            q.min() < 0 or q.max() >= n - 1
            for q, n in zip((xy[1] + s * yy, xy[0] + s * xx), fields["median"].shape, strict=True)
        ):
            raise ValueError("Interpolating outside native support")
        a = {
            k: samples(v.astype("f8"), yy, xx, xy) * doc["slope"] + doc["intercept"]
            for k, v in fields.items()
        }
        b = med[yy, xx]
        published = pub[yy, xx]
        observed = published - b
        grad = np.hypot(*np.gradient(b))
        mid = float(np.median(grad))
        masks = {"all": np.ones_like(b, bool), "weak": grad <= mid, "strong": grad > mid}
        result[part] = dict(
            median_display_gradient=mid,
            methods={
                k: {
                    group: metric(v, a["median"], published, observed, m)
                    for group, m in masks.items()
                }
                for k, v in a.items()
            },
        )
    winner = min(arms, key=lambda k: result["fit"]["methods"][k]["all"]["change_rmse"])
    right = result["check"]["methods"]
    best = right[winner]["all"]["change_rmse"]
    ratios = {k: best / right[k]["all"]["change_rmse"] for k in ["median", "r10h30"]}
    record = dict(
        partitions=result,
        selected=winner,
        check_ratios=ratios,
        advance=all(v <= 0.9 for v in ratios.values()),
        published_nlm_sha256=sha(panels / "nlm.png"),
    )
    write(out, record)
    print(winner, ratios, record["advance"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("predict", "score"))
    for k in ("cache", "pred", "panels", "out"):
        p.add_argument("--" + k, type=Path, required=k == "out")
    a = p.parse_args()
    predict(a.cache, a.out) if a.phase == "predict" else evaluate(a.pred, a.panels, a.out)
