"""Seal within-patch profile predictions on calibration crops and the published plane."""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_axial_raw import DEV, ROOT, SETTINGS, Denoise, digest, sha, stack, verified, write
from pore_figure_gray import E as GRAY
from pore_figure_gray import limits
from pore_patch_stats import decision, score
from pore_slice_noise import md
from pore_window_units import BANDS, H, frac, region
from scipy import ndimage as ndi

E = ROOT / "pores/evidence/patch-profile"
PTX = ROOT / "build/pores/patch-profile.ptx"
ARMS = ("uniform", "binomial")


def direct(im, y, x, binomial, unit):
    # Each search center contributes one intensity and the explicit 25-term distance.
    pad = np.pad(im.astype("f8"), 12, mode="edge")
    all_patches = np.lib.stride_tricks.sliding_window_view(pad[y : y + 25, x : x + 25], (5, 5))
    v = np.array([1, 4, 6, 4, 1]) if binomial else np.ones(5)
    w = np.outer(v, v) / v.sum() ** 2
    dist = ((all_patches - all_patches[10, 10]) ** 2 * w).sum(axis=(2, 3))
    weights = np.exp(-dist[..., None] / (unit * np.array(BANDS)) ** 2)
    return (weights * all_patches[:, :, 2, 2, None]).sum((0, 1)) / weights.sum((0, 1))


def numeric(gpu):
    v = np.random.default_rng(9621).integers(25000, 42001, (1, 23, 29), dtype="u2")
    audit = []
    for b, name in enumerate(ARMS):
        g = gpu(v, 160, weighted=b)[0]
        for y, x in [(0, 0), (12, 14), (20, 27)]:
            expect = direct(v[0], y, x, b, 160)
            err = float(np.max(np.abs(g[:, y, x] - expect)))
            if err > 0.003:
                raise ValueError((name, y, x, err))
            audit.append(
                dict(
                    arm=name, y=y, x=x, cpu=expect.tolist(), cuda=g[:, y, x].tolist(), max_abs=err
                )
            )
    return audit


def paper_raw(cache, gpu, out):
    old = json.loads((GRAY / "predictions.json").read_text())
    par = old["xy0_scale"]
    z = old["z"]
    box = limits(par, (112, 428))
    raw, rs = verified(cache, 2, "raw", False)
    if box[0] < 616 or rs["sha256"] != old["raw"] or z != 447:
        raise ValueError("Wrong paper geometry")
    frames = []
    with ZipFile(raw) as archive:
        for j in range(z - 1, z + 2):
            with Image.open(
                io.BytesIO(archive.read(f"set1sample2raw/set1sample2raw_{j:04d}.tif"))
            ) as im:
                frames.append(np.array(im.crop(box)))
    raw = np.stack(frames)
    m = ndi.median_filter(raw, 3, mode="nearest")[1]
    cut = np.s_[24:-24, 24:-24]
    fields = {}
    if (
        digest(raw) != old["raw_crop"]
        or digest(m[cut]) != old["fields"]["median"]
        or digest(raw[1][cut]) != old["fields"]["original"]
    ):
        raise ValueError("Different median or native source")
    for b, name in enumerate(ARMS):
        vals = gpu(m[None], 1 / SETTINGS[2][0], weighted=b)[0]
        for h, plane in zip(BANDS, vals, strict=True):
            a = plane[cut].copy()
            if not b and digest(a) != old["fields"][f"r10h{h}"]:
                raise ValueError(("Different uniform gray plane", h))
            fields[f"{name}{h}"] = a
    path = out / "fields.npz"
    np.savez_compressed(path, **fields)
    return dict(
        file=sha(path),
        fields={k: digest(v) for k, v in fields.items()},
        gray_seal=sha(GRAY / "predictions.json"),
    )


def predict(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    rec = {}
    previous = json.loads((ROOT / "pores/evidence/window-units/predictions.json").read_text())[
        "specimens"
    ]
    with Denoise(PTX, adaptive=True) as gpu:
        audit = numeric(gpu)
        paper = paper_raw(cache, gpu, out)
        for i in DEV:
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            t = stack(seg, i, "segmented", 160, 164)[:, :, :104]
            m = region(raw, i, True)
            held = region(raw, i, False)
            fits = {}
            chosen = {}
            packed = {}
            baseline = None
            for b, name in enumerate(ARMS):

                def masks(data, i=i, b=b):
                    vals = gpu(data, 1 / SETTINGS[i][0], weighted=b)
                    return [decision(vals[:, j], i, 0)[:, H:-H, H:-H] for j in range(6)]

                fits[name] = {h: score(v, t) for h, v in zip(BANDS, masks(m), strict=True)}
                h = max(BANDS, key=lambda k: frac(fits[name][k]))
                chosen[name] = h
                val = masks(held)
                packed[name] = val[BANDS.index(h)]
                if not b:
                    baseline = md(val[BANDS.index(30)])
                    if (
                        baseline != previous[str(i)]["baseline_h30"]
                        or md(packed[name]) != previous[str(i)]["masks"]["control"]
                    ):
                        raise ValueError("Different uniform mask")
            f = out / f"sample{i}.npz"
            np.savez_compressed(f, **packed)
            rec[i] = dict(
                raw=rs["sha256"],
                segmented=ss["sha256"],
                fit=fits,
                band=chosen,
                file=sha(f),
                masks={k: md(v) for k, v in packed.items()},
                uniform_fixed_h30=baseline,
            )
            print(i, chosen, flush=True)
    write(out / "predictions.json", dict(ptx=sha(PTX), paper=paper, numeric=audit, specimens=rec))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    predict(a.cache, a.out)
