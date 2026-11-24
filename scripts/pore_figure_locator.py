"""Locate a raw CT plane with masked high-pass normalized correlation, no labels."""

import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_axial_raw import sha, verified, write
from scipy import ndimage as ndi
from scipy.fft import irfft2, rfft2

N = 96
SHIFT = 8


def orient(a, k):
    if k & 1:
        a = a[:, ::-1]
    if k & 2:
        a = a[::-1, :]
    if k & 4:
        a = a.T
    return a


def reduce(im):
    a = np.asarray(im.convert("F").resize((N, N), Image.Resampling.BOX), dtype="f8")
    return a - ndi.gaussian_filter(a, 2.5, mode="nearest")


def reference(plane, mask):
    a = plane[mask]
    centered = np.where(mask, plane - a.mean(), 0)
    return (
        rfft2(mask.astype("f8")).conj(),
        rfft2(centered).conj() / np.linalg.norm(centered),
        int(mask.sum()),
    )


def corr(a, m, b, n):
    s = rfft2(a)
    first = irfft2(s * m, s=a.shape)
    second = irfft2(rfft2(a * a) * m, s=a.shape)
    cross = irfft2(s * b, s=a.shape)
    denom = np.sqrt(np.maximum(second - first**2 / n, 1e-30))
    ii = np.arange(-SHIFT, SHIFT + 1) % N
    return (cross / denom)[np.ix_(ii, ii)]


def locate(cache, panels):
    raw, rs = verified(cache, 2, "raw", False)
    overview = panels / "overview.png"
    with Image.open(overview) as im:
        t = reduce(im)
    y, x = (np.arange(N) + 0.5) / N, (np.arange(N) + 0.5) / N
    yy, xx = np.meshgrid(y, x, indexing="ij")
    mask = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.37**2) & ~(
        (xx > 0.57) & (xx < 0.87) & (yy > 0.32) & (yy < 0.63)
    )
    m, b, n = reference(t, mask)
    rows = []
    with ZipFile(raw) as archive:
        names = sorted(
            v
            for v in archive.namelist()
            if v.startswith("set1sample2raw/set1sample2raw_") and v.lower().endswith(".tif")
        )
        for name in names:
            z = int(name[-8:-4])
            with Image.open(io.BytesIO(archive.read(name))) as im:
                if im.size != (980, 1010):
                    raise ValueError("Wrong frame")
                a = reduce(im)
            best = None
            for k in range(8):
                scores = corr(orient(a, k), m, b, n)
                py, px = np.unravel_index(scores.argmax(), scores.shape)
                row = (float(scores[py, px]), k, int(py) - SHIFT, int(px) - SHIFT)
                if best is None or row[0] > best[0]:
                    best = row
            rows.append(dict(z=z, ncc=best[0], orientation=best[1], dy=best[2], dx=best[3]))
            if z % 200 == 0:
                print("raw locator", z, flush=True)
    if len(rows) != len({v["z"] for v in rows}) or len(rows) < 8:
        raise ValueError("Invalid slice inventory")
    top = max(rows, key=lambda r: r["ncc"])
    rival = max((r for r in rows if abs(r["z"] - top["z"]) > 3), key=lambda r: r["ncc"])
    margin = top["ncc"] - rival["ncc"]
    return dict(
        specimen=2,
        source=rs["sha256"],
        overview_sha256=sha(overview),
        native_shape_yx=[1010, 980],
        grid=N,
        max_shift=SHIFT,
        support=n,
        number_of_planes=len(rows),
        peak=top,
        rival=rival,
        margin=margin,
        advance=top["ncc"] >= 0.7 and margin >= 0.08,
        all_planes=rows,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "panels", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    result = locate(a.cache, a.panels)
    write(a.out, result)
    print({k: v for k, v in result.items() if k != "all_planes"}, flush=True)
