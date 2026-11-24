"""Render a prediction-only offline review sheet, then bundle it with the audited full review."""

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import re
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
from audit import decode  # noqa: E402
from bundle_review import write_bundle  # noqa: E402


def extract(folder, i):
    s = f"sample{i}"
    r = json.loads((folder / (s + ".json")).read_text(encoding="utf8"))
    data = (folder / (s + "-review.html")).read_bytes()
    if hashlib.sha256(data).hexdigest() != r["review_sha256"]:
        raise ValueError("Review hash mismatch")
    return json.loads(
        re.search(rb'<script id="data" type="application/json">(.*?)</script>', data, re.S)[1]
    ), r["source"]["raw"]["sha256"]


def uri(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render(c, d, e, i):
    trio = [extract(x, i) for x in (c, d, e)]
    p = trio[2][0]

    def reg(q):
        return (q[1], q[0]["specimen"], q[0]["shape"], q[0]["origin"], q[0]["spacing"])

    if not all(reg(q) == reg(trio[2]) for q in trio):
        raise ValueError("Source geometry differs")
    ox, oy, oz = p["origin"]
    _, ny, nx = p["shape"]
    rows = []
    with (e / f"sample{i}-review-queue.csv").open(newline="", encoding="utf8") as f:
        items = list(csv.DictReader(f))
    for item in items:
        key = item["e_pred_id"]
        z = p["objects"][key]["best"]
        ids = [item["c_peer_id"], item["d_peer_id"], key]
        layers = []
        for (q, _), k in zip(trio, ids, strict=True):
            mask = (
                decode(
                    [seg for seg in q["layers"][z]["objects"] if str(seg[2]) == k], ny * nx
                ).reshape(ny, nx)
                > 0
            )
            layers.append(mask)
        yy, xx = np.nonzero(layers[2])
        y = int(np.clip((yy.min() + yy.max()) // 2 - 32, 0, ny - 64))
        x = int(np.clip((xx.min() + xx.max()) // 2 - 32, 0, nx - 64))
        sl = (slice(y, y + 64), slice(x, x + 64))
        with Image.open(io.BytesIO(base64.b64decode(p["layers"][z]["image"]))) as im:
            raw = np.array(im.convert("RGB"))[sl].copy()
        panels = [("Raw", uri(raw))]
        for j, (method, mask, color) in enumerate(
            zip("CDE", layers, [(58, 187, 213), (236, 177, 62), (255, 117, 102)], strict=True)
        ):
            m = mask[sl]
            image = raw.copy()
            image[m] = 0.35 * image[m] + 0.65 * np.array(color)
            panels.append((f"{method} #{ids[j]}" if ids[j] else f"{method}: no peer", uri(image)))
        figures = "".join(
            f'<figure><img width="96" height="96" alt="{html.escape(label)}" src="{src}"><figcaption>{html.escape(label)}</figcaption></figure>'
            for label, src in panels
        )
        desc = []
        for peer in ("c", "d"):
            desc.append(
                f"{peer.upper()}: "
                + (
                    f"IoU {float(item[peer + '_iou']):.3f}, volume difference {int(item[peer + '_peer_minus_e_voxels']):+d} voxels"
                    if item[peer + "_peer_id"]
                    else "no overlapping eligible peer"
                )
            )
        flag = item["flagged"] == "True"
        rows.append(
            f"<article><h2>#{key} &middot; {item['e_voxels']} voxels</h2><p>{'Flagged' if flag else 'Not flagged'} &middot; {'; '.join(desc)}.</p><div>{figures}</div><small>source z={z + oz}; X=[{x + ox},{x + ox + 64}), Y=[{y + oy},{y + oy + 64})</small></article>"
        )
    return (
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sample {i}: prediction-only review</title><style>body{{max-width:56rem;margin:2rem auto;padding:0 1rem;color:#21372c;background:#faf9f5;font:16px/1.5 system-ui}}article{{border-top:1px solid #b7c7bd;padding:.5rem 0 1rem}}article div{{display:flex;flex-wrap:wrap;gap:1em}}h1,h2{{font-family:Georgia,serif}}figure{{margin:0}}figcaption,small{{font:12px monospace}}img{{image-rendering:pixelated}}</style><h1>Sample {i}: cross-method review</h1><p>{len(items)} eligible E detections, flagged first. All four panels show the same 64&times;64 window on the largest E cross-section. Peer selection and flags use the full 3-D predictions. No reference overlay. An unflagged object can still be wrong; a pore missed by E is not listed.</p><p><a href="sample{i}-cases.html">Check detections and misses against NIST</a> &middot; <a href="sample{i}-review.html">Full slice viewer</a></p>'
        + "".join(rows)
        + "</html>"
    )


def package(c, d, e, i, destination):
    s = f"sample{i}"
    (e / f"{s}-review-queue.html").write_text(render(c, d, e, i), encoding="utf8")
    write_bundle(e, destination, i)
    with ZipFile(destination) as z:
        files = {n: z.read(n) for n in z.namelist() if n != "SHA256SUMS.txt"}
    for suffix in (
        "-overlap.csv",
        "-size-bands.csv",
        "-paired-d-e.csv",
        "-review-queue.csv",
        "-review-queue.html",
    ):
        files[s + suffix] = (e / (s + suffix)).read_bytes()
    files["READ-ME.txt"] += (
        f"Open {s}-review-queue.html to compare C, D and E without a reference overlay. The paired D/E CSV also lists reference pores missed by either or both methods. Never treat a queue flag as an acceptance result.\n".encode(
            "ascii"
        )
    )
    files["SHA256SUMS.txt"] = "".join(
        f"{hashlib.sha256(files[n]).hexdigest()}  {n}\n" for n in sorted(files)
    ).encode("ascii")
    temp = destination.with_suffix(".partial.zip")
    with ZipFile(temp, "w") as z:
        for n, data in sorted(files.items()):
            info = ZipInfo(n, (2026, 11, 24, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 0
            z.writestr(info, data)
    with ZipFile(temp) as z:
        if z.testzip() or any(z.read(k) != v for k, v in files.items()):
            raise ValueError("Archive failed round trip")
    temp.replace(destination)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("c", type=Path)
    ap.add_argument("d", type=Path)
    ap.add_argument("e", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("specimens", nargs="+", type=int, choices=(2, 3, 5))
    a = ap.parse_args()
    for i in a.specimens:
        package(a.c, a.d, a.e, i, a.output / f"candidate-E-sample{i}.zip")
