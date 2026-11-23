"""Locate every eligible match, missed reference and unmatched detection in source coordinates."""

import argparse
import base64
import csv
import html
import io
import json
import re
from pathlib import Path

import numpy as np
from audit import decode, require
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def eligible(labels):
    values, cnt = np.unique(labels, return_counts=True)
    cut = {0}
    for axis in range(3):
        for last in [0, -1]:
            cut.update(np.unique(np.take(labels, last, axis=axis)))
    return {int(v) for v, n in zip(values, cnt, strict=True) if n >= 64 and v not in cut}


def peak(delta, reference):
    active = delta if delta.any() else reference
    z = int(active.sum(axis=(1, 2)).argmax())
    y, x = np.nonzero(active[z])
    k = int(np.argmin((y - y.mean()) ** 2 + (x - x.mean()) ** 2))
    return z, int(y[k]), int(x[k])


def image_url(im):
    f = io.BytesIO()
    im.save(f, format="PNG")
    return "data:image/png;base64," + base64.b64encode(f.getvalue()).decode("ascii")


def export(folder, i):
    stem = f"sample{i}"
    txt = (folder / (stem + "-review.html")).read_text(encoding="utf8")
    data = json.loads(
        re.search(r'<script id="data" type="application/json">(.*?)</script>', txt, re.S)[1]
    )
    del txt
    shape = tuple(data["shape"])
    nz, ny, nx = shape
    ox, oy, oz = data["origin"]
    ids = np.empty(shape, "i4")
    truth = np.empty(shape, bool)
    for z, L in enumerate(data["layers"]):
        ids[z] = decode(L["objects"], ny * nx).reshape(ny, nx)
        truth[z] = decode(L["reference"], ny * nx, True).reshape(ny, nx)
    ref, _ = ndi.label(truth)
    predids = eligible(ids)
    refids = eligible(ref)
    with (folder / (stem + "-matches.csv")).open(newline="") as f:
        pairs = list(csv.DictReader(f))
    cases = []
    for pair in pairs:
        p, r = int(pair["pred_id"]), int(pair["ref_scipy_label"])
        predids.remove(p)
        refids.remove(r)
        cases.append(("matched", p, r, float(pair["signed_volume_error_pct"])))
    cases.sort(key=lambda c: -abs(c[3]))
    cases = (
        [("missed reference", None, r, None) for r in sorted(refids)]
        + [("unmatched detection", p, None, None) for p in sorted(predids)]
        + cases
    )
    reports = json.loads((folder / (stem + ".json")).read_text(encoding="utf8"))["objects"]
    require(
        len(refids) + len(pairs) == reports["eligible_reference"]
        and len(predids) + len(pairs) == reports["eligible_prediction"],
        "Eligibility disagreement",
    )
    # Compute each object's bounding box once instead of allocating one full mask per object.
    labs, _ = ndi.label(ids != 0)
    ps = {}
    pboxes = ndi.find_objects(labs)
    rboxes = ndi.find_objects(ref)
    for key, v in data["objects"].items():
        idx = tuple(int(v["row"]["seed_" + k]) for k in "zyx")
        ps[int(key)] = pboxes[labs[idx] - 1]
    records = []
    rows = []
    for status, p, r, err in cases:
        boxes = ([ps[p]] if p is not None else []) + ([rboxes[r - 1]] if r is not None else [])
        box = tuple(
            slice(min(b[a].start for b in boxes), max(b[a].stop for b in boxes)) for a in range(3)
        )
        sh = ids[box].shape
        A = ids[box] == p if p is not None else np.zeros(sh, bool)
        B = ref[box] == r if r is not None else np.zeros(sh, bool)
        dz, dy, dx = peak(A ^ B, A | B)
        z, y, x = (v + b.start for v, b in zip((dz, dy, dx), box, strict=True))
        x0 = max(0, min(x - 24, nx - 48))
        y0 = max(0, min(y - 24, ny - 48))
        window = (x0, y0, x0 + 48, y0 + 48)
        raw = (
            Image.open(io.BytesIO(base64.b64decode(data["layers"][z]["image"])))
            .convert("RGB")
            .crop(window)
        )
        rgb = np.array(raw)
        arr = rgb.astype(float)
        a = ids[z, y0 : y0 + 48, x0 : x0 + 48] == p if p is not None else np.zeros((48, 48), bool)
        b = ref[z, y0 : y0 + 48, x0 : x0 + 48] == r if r is not None else np.zeros((48, 48), bool)
        for mask, color in [
            (a & ~b, [255, 127, 34]),
            (b & ~a, [20, 215, 226]),
            (a & b, [255, 255, 255]),
        ]:
            rgb[mask] = 0.25 * arr[mask] + 0.75 * np.array(color)
        flagged = Image.fromarray(rgb)
        tx, ty = x - x0, y - y0
        pen = ImageDraw.Draw(flagged)
        pen.rectangle((tx - 2, ty - 2, tx + 2, ty + 2), outline=(240, 75, 115))
        filename = f"set1sample{i}raw_{oz + z:04d}.tif"
        rec = (status, p, r, err, ox + x, oy + y, oz + z, filename)
        records.append(rec)
        rows.append(
            f'<tr><td>{html.escape(status)}<br>prediction {p or "—"} / ref {r or "—"}</td><td>{"—" if err is None else f"{err:+.2f}%"}</td><td><img src="{image_url(raw)}" alt="Raw crop"> <img src="{image_url(flagged)}" alt="Component discrepancy overlay"></td><td>{ox + x}, {oy + y}, {oz + z}<br>{filename}</td></tr>'
        )
    with (folder / (stem + "-cases.csv")).open("w", newline="", encoding="utf8") as out:
        w = csv.writer(out)
        w.writerow(
            [
                "status",
                "pred_id",
                "reference_label",
                "signed_volume_error_pct",
                "source_x",
                "source_y",
                "source_z",
                "source_tiff",
            ]
        )
        w.writerows(records)
    result = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Case index</title><style>body{font:16px/1.5 system-ui;margin:2rem auto;padding:0 1rem;max-width:72rem;color:#193c33}h1{font:2.5rem Georgia,serif}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid #baccc1;padding:.6rem;font-size:.9rem}img{width:78px;image-rendering:pixelated}section{overflow-x:auto}a{color:#086453}</style>"""
    result += (
        f'<h1>Sample {i} · discrepancy index</h1><p>{len(pairs)} matched, {len(refids)} missed references, {len(predids)} unmatched detections. Only interior objects ≥64 voxels. Source coordinates are zero-based. Volume difference is signed (prediction vs. reference).</p><p>Each pair shows its worst disagreement slice (most XOR voxels). The square marks an actual disagreement voxel, or an object voxel for identical pairs. Orange: prediction only; cyan: reference only; white: both. Only the listed object(s) are colored. <a href="{stem}-review.html">Open the full slice viewer</a>.</p><section><table><thead><tr><th>Case</th><th>Volume error</th><th>Raw / overlay</th><th>Source X, Y, Z</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></section><footer>Derived from Kim et al., NIST, DOI 10.18434/M32162. No physical dimensional certification.</footer>"
    )
    (folder / (stem + "-cases.html")).write_text(result, encoding="utf8")
    return len(records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("specimens", type=int, nargs="+")
    args = ap.parse_args()
    for i in args.specimens:
        print(i, export(args.folder, i))
