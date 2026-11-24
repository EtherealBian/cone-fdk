"""Score only publicly sealed patch-profile fields, with a known median-change control."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pore_axial_raw import DEV, digest, sha, stack, verified, write
from pore_figure_audit import render
from pore_figure_gray import RANGES, REG, metric
from pore_patch_profile import ARMS, GRAY, E
from pore_patch_stats import score
from pore_slice_noise import md
from pore_window_units import BANDS


def read(p):
    return json.loads(p.read_text())


def checked(path, checksum):
    if sha(path) != checksum:
        raise ValueError(("Changed file", path))
    return path


def paper(pred, panels, rec):
    locked = read(checked(GRAY / "predictions.json", rec["gray_seal"]))
    r = read(checked(REG, locked["registration"]))
    prior = read(GRAY / "check.json")
    images = {}
    for key, h in {**r["panels"], "nlm": prior["published_nlm_sha256"]}.items():
        with Image.open(checked(panels / f"{key}.png", h)) as im:
            images[key] = np.array(im, dtype="f8")
    with np.load(checked(GRAY / "fields.npz", locked["file"]), allow_pickle=False) as z:
        sources = {k: z[k] for k in ("original", "median")}
    with np.load(checked(pred / "fields.npz", rec["file"]), allow_pickle=False) as z:
        fields = {k: z[k] for k in z.files}
    if {k: digest(v) for k, v in fields.items()} != rec["fields"]:
        raise ValueError("Different fields")
    results = {}
    for name, (left, right) in RANGES.items():
        yy, xx = np.mgrid[80:424, left:right]
        raw, med, nlm = [images[k][yy, xx] for k in ("raw", "median", "nlm")]
        g = np.hypot(*np.gradient(med))
        cut = np.median(g)
        masks = {"all": np.ones_like(med, bool), "weak": g <= cut, "strong": g > cut}
        base = render(sources["median"], yy, xx, locked)
        origin = render(sources["original"], yy, xx, locked)
        vals = {k: render(v, yy, xx, locked) for k, v in fields.items()}
        measures = {
            k: {g: metric(v, base, nlm, nlm - med, m) for g, m in masks.items()}
            for k, v in vals.items()
        }
        control = {
            k: {g: metric(v, origin, med, med - raw, m) for g, m in masks.items()}
            for k, v in [("no_median", origin), ("native_median", base)]
        }
        for h in BANDS:
            if (
                abs(
                    measures[f"uniform{h}"]["all"]["change_rmse"]
                    - prior["partitions"][name]["methods"][f"r10h{h}"]["all"]["change_rmse"]
                )
                > 1e-10
            ):
                raise ValueError("Published uniform gray result changed")
        results[name] = dict(methods=measures, median_control=control)
    chosen = {
        k: min(BANDS, key=lambda h: results["fit"]["methods"][f"{k}{h}"]["all"]["change_rmse"])
        for k in ARMS
    }
    a, b = [results["check"]["methods"][f"{k}{chosen[k]}"]["all"]["change_rmse"] for k in ARMS]
    return dict(parts=results, selected=chosen, right_ratio=b / a, advance=b / a <= 0.9)


def run(cache, pred, panels):
    seal = read(E / "predictions.json")
    if read(pred / "predictions.json") != seal or set(seal["specimens"]) != set("235"):
        raise ValueError("Invalid input seal")
    fig = paper(pred, panels, seal["paper"])
    scans = {}
    for i in DEV:
        r = seal["specimens"][str(i)]
        seg, ss = verified(cache, i, "segmented", False)
        if ss["sha256"] != r["segmented"]:
            raise ValueError("Wrong check source")
        with np.load(checked(pred / f"sample{i}.npz", r["file"]), allow_pickle=False) as f:
            a, b = [f[k] for k in ARMS]
        if [md(a), md(b)] != [r["masks"][k] for k in ARMS]:
            raise ValueError("Changed masks")
        t = stack(seg, i, "segmented", 176, 184)[:, :, 152:]
        x, y = score(a, t), score(b, t)
        delta = {k: y[k] - x[k] for k in ("dice", "precision", "recall")}
        hist = np.bincount((4 * t.astype("u1") + 2 * a + b).ravel(), minlength=8)
        scans[i] = dict(
            uniform=x,
            binomial=y,
            delta=delta,
            truth_uniform_binomial=hist.tolist(),
            advance=delta["dice"] >= 0.005 and min(delta["precision"], delta["recall"]) >= -0.005,
        )
    return dict(
        paper=fig,
        specimens=scans,
        advance=fig["advance"] and all(v["advance"] for v in scans.values()),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("pred", "cache", "panels", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    d = run(a.cache, a.pred, a.panels)
    write(a.out, d)
    print(
        d["paper"]["selected"],
        d["paper"]["right_ratio"],
        [(i, r["delta"]) for i, r in d["specimens"].items()],
        d["advance"],
    )
