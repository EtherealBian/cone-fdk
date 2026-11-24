"""Verify raw-panel registration on withheld pixels before using a published intermediate."""

import argparse
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_axial_raw import sha, verified, write
from pore_figure_locator import orient
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.signal import correlate


def samples(im, rows, cols, pars):
    x, y, s = pars
    return ndi.map_coordinates(
        im, [y + s * rows, x + s * cols], order=1, mode="nearest", prefilter=False
    )


def ncc(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else 0.0


def coarse(image, panel):
    # Approximate origin of the dashed box in the printed raw overview, with a
    # generous search margin. The second, median panel never enters registration.
    best = (-np.inf, None)
    oy, ox = 80, 20
    for scale in np.linspace(0.38, 0.52, 29):
        fy = oy + np.arange(0, int((424 - oy) * scale)) / scale
        fx = ox + np.arange(0, int((216 - ox) * scale)) / scale
        by, bx = np.meshgrid(fy, fx, indexing="ij")
        patch = ndi.map_coordinates(panel, [by, bx], order=1)
        patch -= patch.mean()
        ny, nx = patch.shape
        y0, x0 = int(310 + scale * oy), int(535 + scale * ox)
        arr = image[y0 : int(418 + scale * oy) + ny, x0 : int(650 + scale * ox) + nx]
        one = np.ones(patch.shape)
        sm = correlate(arr, one, "valid", method="fft")
        sq = correlate(arr**2, one, "valid", method="fft")
        dots = correlate(arr, patch, "valid", method="fft")
        sc = dots / (np.linalg.norm(patch) * np.sqrt(np.maximum(sq - sm**2 / patch.size, 1e-12)))
        yy, xx = np.unravel_index(sc.argmax(), sc.shape)
        if sc[yy, xx] > best[0]:
            best = float(sc[yy, xx]), [x0 + xx - scale * ox, y0 + yy - scale * oy, scale]
    return best


def stats(a, b):
    span = float(np.percentile(b, 95) - np.percentile(b, 5))
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    return dict(
        ncc=ncc(a, b), rmse_display=rms, p95_p5_display=span, nrmse=rms / span, pixels=int(b.size)
    )


def run(cache, panels, loc):
    d = json.loads(loc.read_text())
    r = d["peak"]
    if (
        not d["advance"]
        or d["overview_sha256"] != sha(panels / "overview.png")
        or r["orientation"] != 0
    ):
        raise ValueError("Requires the verified identity-orientation locator")
    raw, rs = verified(cache, 2, "raw", False)
    if rs["sha256"] != d["source"]:
        raise ValueError("Changed source")
    with ZipFile(raw) as z:
        planes = []
        for j in range(r["z"] - 1, r["z"] + 2):
            with Image.open(
                io.BytesIO(z.read(f"set1sample2raw/set1sample2raw_{j:04d}.tif"))
            ) as im:
                planes.append(orient(np.asarray(im), 0))
    v = np.stack(planes)
    image = v[1].astype("f8")
    med = ndi.median_filter(v, size=3)[1].astype("f8")
    with Image.open(panels / "raw.png") as im:
        panel = np.asarray(im, dtype="f8")
    with Image.open(panels / "median.png") as im:
        other = np.asarray(im, dtype="f8")
    if panel.shape != (448, 448) or panel.shape != other.shape:
        raise ValueError("Unexpected panel geometry")
    rough, initial = coarse(image, panel)
    by, bx = np.mgrid[80:424:2, 20:216:2]
    truth = panel[by, bx]

    def cost(q):
        return -ncc(samples(image, by, bx, q), truth)

    opt = minimize(
        cost,
        initial,
        method="Nelder-Mead",
        options={
            "initial_simplex": np.array(
                [
                    initial,
                    np.add(initial, [0.5, 0, 0]),
                    np.add(initial, [0, 0.5, 0]),
                    np.add(initial, [0, 0, 0.002]),
                ]
            ),
            "xatol": 1e-7,
            "fatol": 1e-11,
            "maxiter": 400,
        },
    )
    if not opt.success:
        raise ValueError("Registration failed to converge")
    par = opt.x
    B = samples(image, by, bx, par)
    slope, intercept = np.linalg.lstsq(
        np.stack([B.ravel(), np.ones(B.size)], axis=1), truth.ravel(), rcond=None
    )[0]
    yy, xx = np.mgrid[80:424, 232:428]
    a = stats(samples(image, yy, xx, par) * slope + intercept, panel[yy, xx])
    b = stats(samples(med, yy, xx, par) * slope + intercept, other[yy, xx])
    b["nrmse_using_raw_panel_span"] = b["rmse_display"] / a["p95_p5_display"]
    return dict(
        specimen=2,
        z=r["z"],
        raw=rs["sha256"],
        locator_sha256=sha(loc),
        panels={k: sha(panels / f"{k}.png") for k in ("raw", "median")},
        coarse_left_ncc=rough,
        fit_left_ncc=-opt.fun,
        source_xy0_scale=par.tolist(),
        intensity_slope=float(slope),
        intensity_intercept=float(intercept),
        test_right_raw=a,
        test_right_median=b,
        advance=bool(
            a["ncc"] >= 0.90
            and b["ncc"] >= 0.90
            and max(b["nrmse"], b["nrmse_using_raw_panel_span"]) <= 0.12
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for s in ("cache", "panels", "locator", "out"):
        p.add_argument("--" + s, type=Path, required=True)
    a = p.parse_args()
    v = run(a.cache, a.panels, a.locator)
    write(a.out, v)
    print(v)
