"""Compare a fixed off-ROI prediction, the NIST labels and (only after seal) a print crop."""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_axial_raw import ROOT, SETTINGS, Denoise, digest, sha, verified, write
from pore_figure_gray import RANGES, REG, limits
from pore_figure_gray import E as GRAY
from pore_patch_stats import decision
from pore_window_units import BANDS, PTX
from scipy import ndimage as ndi

E = ROOT / "pores/evidence/figure-binary"


def coords(par, split):
    ox, oy, s = par
    lo, hi = RANGES[split]
    yy, xx = np.mgrid[0:1010, 0:980]
    u = np.floor((xx - ox) / s + 0.5).astype("i4")
    v = np.floor((yy - oy) / s + 0.5).astype("i4")
    keep = (u >= lo) & (u < hi) & (v >= 80) & (v < 424)
    return np.stack((xx[keep], yy[keep], u[keep], v[keep]), axis=1).astype("<i4")


def read_tif(archive, kind, z, box):
    stem = "set1sample2" + kind
    with Image.open(io.BytesIO(archive.read(f"{stem}/{stem}_{z:04d}.tif"))) as im:
        if im.size != (980, 1010):
            raise ValueError("Different source shape")
        return np.asarray(im.crop(box))


def source(cache, out):
    d = json.loads((GRAY / "predictions.json").read_bytes())
    par = d["xy0_scale"]
    box = limits(par, (112, 428))
    x0, y0, _, _ = box
    raw, rs = verified(cache, 2, "raw", False)
    seg, ss = verified(cache, 2, "segmented", False)
    z = d["z"]
    if sha(REG) != d["registration"] or x0 < 616 or z != 447 or rs["sha256"] != d["raw"]:
        raise ValueError("Different source")
    with ZipFile(raw) as a:
        v = np.stack([read_tif(a, "raw", j, box) for j in (z - 1, z, z + 1)])
    if digest(v) != d["raw_crop"]:
        raise ValueError("Wrong raw region")
    med = ndi.median_filter(v, 3, mode="nearest")[1]
    with Denoise(PTX, adaptive=True) as gpu:
        field = gpu(med[None], 1 / SETTINGS[2][0], weighted=0, noise=2)[:, BANDS.index(30)]
    fieldhash = digest(field[0, 24:-24, 24:-24])
    if fieldhash != d["fields"]["r10h30"]:
        raise ValueError("Different source denoiser")
    pred = decision(field, 2, 0)[0]
    with ZipFile(seg) as a:
        archive = read_tif(a, "segmented", z, box)
    if not np.isin(archive, [0, 255]).all():
        raise ValueError("Invalid label encoding")
    packed = {}
    spans = {}
    for split in RANGES:
        xy = coords(par, split)
        x, y = xy[:, :2].T
        if (
            min((x - x0).min(), (y - y0).min(), (box[2] - 1 - x).min(), (box[3] - 1 - y).min())
            < 18
        ):
            raise ValueError("Missing true source halo")
        packed[split + "_xyuv"] = xy
        packed[split + "_fixed"] = pred[y - y0, x - x0]
        packed[split + "_archive"] = archive[y - y0, x - x0] == 0
        spans[split] = [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]
    out.mkdir(parents=True, exist_ok=False)
    f = out / "native.npz"
    np.savez_compressed(f, **packed)
    write(
        out / "source.json",
        dict(
            raw=rs["sha256"],
            segmented=ss["sha256"],
            registration=d["registration"],
            gray_prediction=sha(GRAY / "predictions.json"),
            field_sha256=fieldhash,
            raw_support=box,
            bounds=spans,
            z=z,
            xy0_scale=par,
            file=sha(f),
            arrays={k: digest(v) for k, v in packed.items()},
        ),
    )


def metrics(a, b):
    # First array is reference, second is prediction.
    tn, fp, fn, tp = map(int, np.bincount((2 * a + b).ravel(), minlength=4))

    def div(n, d):
        return n / d if d else None

    return dict(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        dice=div(2 * tp, 2 * tp + fp + fn),
        precision=div(tp, tp + fp),
        recall=div(tp, tp + fn),
    )


def score(pred, panel):
    d = json.loads((E / "source.json").read_bytes())
    f = pred / "native.npz"
    if (
        json.loads((pred / "source.json").read_bytes()) != d
        or sha(f) != d["file"]
        or sha(REG) != d["registration"]
    ):
        raise ValueError("Unknown source seal")
    with Image.open(panel) as im:
        if im.size != (448, 448) or im.mode != "L":
            raise ValueError("Unexpected final panel")
        p = np.asarray(im)
    rows = {}
    with np.load(f, allow_pickle=False) as z:
        if {k: digest(z[k]) for k in z.files} != d["arrays"]:
            raise ValueError("Changed native observations")
        for split in RANGES:
            xy = z[split + "_xyuv"]
            a = z[split + "_archive"]
            b = z[split + "_fixed"]
            g = p[xy[:, 3], xy[:, 2]]
            c = g < 128
            m = (g >= 16) & (g <= 239)
            cells = 4 * a + 2 * b + c
            d0 = metrics(a, c)
            supported = all(min(q.sum(), len(q) - q.sum()) >= 32 for q in (a, c))
            rows[split] = dict(
                count=len(a),
                archive_fixed_printed=np.bincount(cells, minlength=8).tolist(),
                intermediate_tones=np.bincount(cells[m], minlength=8).tolist(),
                archive_reference_printed=d0,
                archive_reference_fixed=metrics(a, b),
                printed_reference_fixed=metrics(c, b),
                sufficient=supported,
                concordant=bool(
                    supported
                    and d0["dice"] >= 0.97
                    and min(d0["precision"], d0["recall"]) >= 0.95
                ),
            )
    return dict(
        panel_sha256=sha(panel),
        partitions=rows,
        concordant=all(r["concordant"] for r in rows.values()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("source", "score"))
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--pred", type=Path, default=E)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    source(a.cache, a.out) if a.phase == "source" else write(a.out, score(a.pred, a.panel))
