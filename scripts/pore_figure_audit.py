"""Independent replay of printed-gray summaries; no paper pixels in the output."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "pores/evidence"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(field, yy, xx, d):
    x, y, s = d["xy0_scale"]
    ox, oy = d["field_origin_xy"]
    xf = (x - ox) + s * xx
    yf = (y - oy) + s * yy
    u = xf.astype(int)
    v = yf.astype(int)
    wx = xf - u
    wy = yf - v
    if (
        min(xf.min(), yf.min()) < 0
        or u.max() + 1 >= field.shape[1]
        or v.max() + 1 >= field.shape[0]
    ):
        raise ValueError("Outside sealed plane")
    val = (1 - wy) * ((1 - wx) * field[v, u] + wx * field[v, u + 1]) + wy * (
        (1 - wx) * field[v + 1, u] + wx * field[v + 1, u + 1]
    )
    return val * d["slope"] + d["intercept"]


def check(panels):
    loc, gray = ROOT / "figure-locator", ROOT / "figure-gray"
    d = json.loads((gray / "predictions.json").read_text())
    expected = json.loads((gray / "check.json").read_text())
    r = json.loads((loc / "registration.json").read_text())
    locator = json.loads((loc / "locator.json").read_text())
    if (
        file_sha(loc / "registration.json") != d["registration"]
        or file_sha(loc / "locator.json") != r["locator_sha256"]
        or not (r["advance"] and locator["advance"] and d["raw"] == r["raw"] == locator["source"])
    ):
        raise ValueError("Broken locator chain")
    for k, h in dict(
        overview=locator["overview_sha256"], nlm=expected["published_nlm_sha256"], **r["panels"]
    ).items():
        if file_sha(panels / f"{k}.png") != h:
            raise ValueError(f"Wrong paper crop: {k}")
    if file_sha(gray / "fields.npz") != d["file"]:
        raise ValueError("Different source fields")
    with np.load(gray / "fields.npz", allow_pickle=False) as fp:
        fields = {k: fp[k] for k in fp.files}
    with Image.open(panels / "median.png") as im:
        m = np.asarray(im, dtype="f8")
    with Image.open(panels / "nlm.png") as im:
        n = np.asarray(im, dtype="f8")
    acc = {}
    count = 0
    worst = 0.0
    for split, (low, high) in {"fit": (112, 216), "check": (232, 428)}.items():
        yy, xx = np.mgrid[80:424, low:high]
        b = m[yy, xx]
        target = n[yy, xx]
        diff = target - b
        g = np.hypot(*np.gradient(b))
        mid = np.median(g)
        base = render(fields["median"], yy, xx, d)
        acc[split] = {}
        for k, a in fields.items():
            pred = render(a, yy, xx, d)
            err = pred - base - diff
            for group, mask in [
                ("all", np.ones_like(b, bool)),
                ("weak", g <= mid),
                ("strong", g > mid),
            ]:
                p = pred[mask]
                t = target[mask]
                e = err[mask]
                scores = dict(
                    count=p.size,
                    change_rmse=np.linalg.norm(e) / np.sqrt(p.size),
                    change_bias=e.mean(),
                    absolute_rmse=np.linalg.norm(p - t) / np.sqrt(p.size),
                    absolute_ncc=np.corrcoef(p, t)[0, 1],
                )
                ref = expected["partitions"][split]["methods"][k][group]
                for name, value in scores.items():
                    delta = abs(value - ref[name])
                    worst = max(worst, delta)
                    count += 1
                    if not np.isfinite(delta) or delta > 2e-10:
                        raise ValueError((split, k, group, name, value, ref[name]))
                if group == "all":
                    acc[split][k] = float(scores["change_rmse"])
    chosen = min((k for k in fields if k.startswith("r")), key=acc["fit"].get)
    ratios = {k: acc["check"][chosen] / acc["check"][k] for k in ("median", "r10h30")}
    if (
        chosen != expected["selected"]
        or any(abs(v - expected["check_ratios"][k]) > 2e-10 for k, v in ratios.items())
        or all(v <= 0.9 for v in ratios.values()) != expected["advance"]
    ):
        raise ValueError("Selection or gate changed")
    return dict(
        measurements_checked=count,
        maximum_absolute_difference=float(worst),
        selected=chosen,
        ratios=ratios,
        paper_pixels_exported=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    d = check(a.panels)
    with a.out.open("x", encoding="utf8", newline="\n") as f:
        json.dump(d, f, indent=2, allow_nan=False)
        f.write("\n")
    print(d)
