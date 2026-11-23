"""No gain/offset fitting. All comparisons use full volumes in global physical coordinates."""

import argparse
import hashlib
import json
import statistics

import numpy as np
from common import DATA, REPORT, ROOT, Geometry, astra_run, difference, reconstruct
from phantom import CYL, SPHERE, generate
from PIL import Image, ImageDraw


def metrology(path, n, fov):
    a = np.memmap(path, "f4", "r", shape=(n, n, n))
    step = fov / n
    t = (np.arange(n) + 0.5 - n / 2) * step
    y, x = np.meshgrid(t, t, indexing="ij")
    img = a[n // 2]
    rad = x * x + y * y
    bulk = rad < 9**2
    holes = []
    for cx, cy, r, *_ in CYL[1:]:
        rr = (x - cx) ** 2 + (y - cy) ** 2
        bulk &= rr > (r + 1.3) ** 2
        mask = (rr < (r + 0.95) ** 2) & (img < 0.02)
        assert mask.any(), "no hole"
        X = x[mask].mean()
        Y = y[mask].mean()
        holes.append(
            {
                "measured_xy_mm": [float(X), float(Y)],
                "center_error_mm": float(np.hypot(X - cx, Y - cy)),
                "diam_error_mm": float(abs(2 * np.sqrt(mask.sum() * step**2 / np.pi) - 2 * r)),
            }
        )
    sz = int(round(SPHERE[2] / step + (n - 1) / 2))
    core = (x - SPHERE[0]) ** 2 + (y - SPHERE[1]) ** 2 < 0.75**2
    return {
        "bulk_rel_error": float(abs(img[bulk].mean() / 0.04 - 1)),
        "outside_mean": float(img[(rad > 12**2) & (rad < 14**2)].mean()),
        "holes": holes,
        "inclusion_mean": float(a[sz][core].mean()),
    }


def png_grid(panels, path, limits=None, subtitle=""):
    allpix = np.concatenate([a.ravel() for _, a in panels])
    lo, hi = limits or np.percentile(allpix, [2, 99])
    size = 310
    w = len(panels) * size
    canvas = Image.new("RGB", (w, size + 78), "#15202a")
    d = ImageDraw.Draw(canvas)
    for i, (name, a) in enumerate(panels):
        im = Image.fromarray(np.uint8(np.clip((a - lo) / (hi - lo), 0, 1) * 255)).resize(
            (size, size), Image.Resampling.BILINEAR
        )
        canvas.paste(im, (i * size, 34))
        d.text((i * size + 9, 10), name, fill="white")
    d.text(
        (10, size + 46),
        subtitle or f"Shared display window: [{lo:.4g}, {hi:.4g}] /mm",
        fill="#c6e3e5",
    )
    canvas.save(path)


def summary(runs):
    return {
        k: statistics.median(x[k] for x in runs)
        for k in runs[0]
        if type(runs[0][k]) in (int, float)
    }


def gated(condition, msg):
    if not condition:
        raise AssertionError(msg)


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run_phantom():
    scan = DATA / "phantom"
    generate(scan)
    allrows = {}
    out = DATA / "native.f32"
    ref = DATA / "reference.f32"
    repeated = DATA / "repeat.f32"
    side = DATA / "slab.f32"
    for n in (256, 512):
        own = []
        baseline = []
        # Warm each implementation and test byte determinism on the same logical input.
        reconstruct(scan, repeated, n, 32)
        astra_run(scan, ref, n, 32)
        for rep in range(3):
            own.append(reconstruct(scan, out, n, 32))
            baseline.append(astra_run(scan, ref, n, 32))
        hashes = {k: sha256(p) for k, p in [("native", out), ("warmup", repeated)]}
        gated(hashes["native"] == hashes["warmup"], "repeat SHA256 mismatch")
        same = difference(out, repeated, n)
        delta = difference(out, ref, n)
        geo = metrology(out, n, 32)
        slab = reconstruct(scan, side, n, 32, 19 if n == 256 else 143, 11)
        split = difference(out, side, n)
        a = np.memmap(out, "f4", "r", shape=(n, n, n))
        b = np.memmap(ref, "f4", "r", shape=a.shape)
        png_grid(
            [
                ("Native CUDA / analytic phantom", a[n // 2]),
                ("ASTRA / same geometry", b[n // 2]),
                ("XZ / off-plane inclusion", a[:, n // 2, :]),
            ],
            REPORT / f"phantom-{n}.png",
            (0, 0.075),
        )
        del a, b
        row = dict(
            native_runs=own,
            astra_runs=baseline,
            native_median=summary(own),
            astra_median=summary(baseline),
            sha256=hashes,
            repeat=same,
            astra_difference=delta,
            metrology=geo,
            limited_run=slab,
            limited_difference=split,
        )
        allrows[str(n)] = row
        (REPORT / "phantom.json").write_text(json.dumps(allrows, indent=2))
        print(
            n,
            json.dumps(
                {k: row[k] for k in ("astra_difference", "metrology", "limited_difference")}
            ),
            flush=True,
        )
        gated(same["max_abs"] == 0, "determinism")
        gated(delta["rel_l2"] < 0.01, "ASTRA mismatch")
        gated(
            slab["slabs"] > 1 and slab["peak_owned_bytes"] <= slab["budget_bytes"],
            "budget",
        )
        gated(split["rel_l2"] <= 1e-5, "slab mismatch")
        gated(
            geo["bulk_rel_error"] < 0.03
            and abs(geo["outside_mean"]) < 0.0015
            and geo["inclusion_mean"] > 0.063,
            "attenuation",
        )
        gated(
            all(h["center_error_mm"] < 0.4 and h["diam_error_mm"] < 0.6 for h in geo["holes"]),
            "hole geometry",
        )
    a = []
    for batch in (1, 16):
        a.append(reconstruct(scan, out, 256, 32, 768, batch))
    (REPORT / "batch-profile.json").write_text(json.dumps(a, indent=2))
    print("phantom and scaling gates passed", flush=True)


def materialize(name, folder=None):
    import shutil

    folder = folder or ROOT / "fixtures/walnut-s8-b8"
    scan = DATA / ("walnut-" + name)
    raw = scan.with_suffix(".f32")
    geom = folder / (name + ".geom")
    g = Geometry.read(geom)
    with np.load(folder / (name + ".npz")) as data:
        x = data["views"]
        assert x.shape == (g.a, g.v, g.u) and x.dtype == np.float32 and np.isfinite(x).all()
        x.tofile(raw)
    shutil.copyfile(geom, scan.with_suffix(".geom"))
    return scan


def run_walnut():
    scans = [materialize(x) for x in ("uncorrected", "corrected")]
    n = 256
    fov = 40.0
    vols = [DATA / (p.name + "-256.f32") for p in scans]
    results = [reconstruct(s, v, n, fov) for s, v in zip(scans, vols)]
    ref = DATA / "walnut-astra.f32"
    base = astra_run(scans[1], ref, n, fov)
    diff = difference(vols[1], ref, n)
    xs = [np.memmap(p, "f4", "r", shape=(n, n, n)) for p in vols + [ref]]
    imgs = [x[n // 2] for x in xs]
    png_grid(
        list(
            zip(
                (
                    "No rotation-center correction",
                    "Corrected: -5 native pixels",
                    "ASTRA / corrected",
                ),
                imgs,
            )
        ),
        REPORT / "walnut.png",
    )
    png_grid(
        [
            ("Axial / corrected", imgs[1]),
            ("Coronal / corrected", xs[1][:, n // 2, :]),
            ("Sagittal / corrected", xs[1][:, :, n // 2]),
        ],
        REPORT / "walnut-orthogonal.png",
    )
    d = difference(vols[0], vols[1], n)
    res = dict(
        native=results,
        corrected_sha256=sha256(vols[1]),
        astra=base,
        corrected_vs_astra=diff,
        uncorrected_vs_corrected=d,
        views=Geometry.read(scans[1].with_suffix(".geom")).a,
        fov_mm=fov,
        n=n,
    )
    (REPORT / "walnut.json").write_text(json.dumps(res, indent=2))
    gated(diff["rel_l2"] < 0.02, "real ASTRA mismatch")
    gated(d["rel_l2"] > 0.001, "axis correction had no effect")
    gated(np.ptp(imgs[1]) > 0.005, "blank walnut")
    print("walnut passed:", diff, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("suite", choices=["phantom", "walnut", "all"])
    a = p.parse_args()
    REPORT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    if a.suite != "walnut":
        run_phantom()
    if a.suite != "phantom":
        run_walnut()
