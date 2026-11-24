"""Render one registered C/D specimen-3 discrepancy from the two offline reviews."""

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pores"))
from audit import decode  # noqa: E402
from validate import sha  # noqa: E402


def load(folder):
    path = folder / "sample3-review.html"
    report = json.loads((folder / "sample3.json").read_text(encoding="utf8"))
    if sha(path) != report["review_sha256"]:
        raise ValueError("Review checksum mismatch")
    d = json.loads(
        re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            path.read_text(encoding="utf8"),
            re.S,
        )[1]
    )
    nz, ny, nx = d["shape"]
    layers = d["layers"]
    return (
        np.stack([decode(layer["objects"], ny * nx).reshape(ny, nx) for layer in layers]),
        np.stack([decode(layer["reference"], ny * nx, True).reshape(ny, nx) for layer in layers]),
        layers,
        d["origin"],
    )


def render(c, d, out):
    pc, ref, cl, origin = load(c)
    pd, refd, dl, od = load(d)
    np.testing.assert_array_equal(ref, refd)
    if od != origin or any(a["image"] != b["image"] for a, b in zip(cl, dl, strict=True)):
        raise ValueError("Source images differ")
    labels, _ = ndi.label(ref)
    first = labels == 121
    second = labels == 127
    join = pc == 3405094
    d1 = pd == 3404321
    d2 = pd == 3597858
    if [int(m.sum()) for m in (first, second, join, d1, d2)] != [296, 128, 436, 278, 108]:
        raise ValueError("Unexpected component identity")

    # Prefer an axial raw slice intersecting both reference pores and containing their C connection.
    def rank(z):
        u = first[z]
        v = second[z]
        cc, _ = ndi.label(join[z])
        shared = (set(cc[u]) - {0}) & (set(cc[v]) - {0})
        return bool(shared), min(int(u.sum()), int(v.sum())), int((u | v).sum()), -z

    z = max(range(first.shape[0]), key=rank)
    coords = np.nonzero(first | second | join | d1 | d2)
    cy, cx = [int((t.min() + t.max()) // 2) for t in coords[1:]]
    half = 14
    box = (slice(cy - half, cy + half + 1), slice(cx - half, cx + half + 1))
    x = origin[0] + cx - half
    y = origin[1] + cy - half
    src_z = origin[2] + z
    raw = np.array(Image.open(io.BytesIO(base64.b64decode(cl[z]["image"]))).convert("RGB"))[box]
    sea = (40, 187, 255)
    gold = (255, 187, 64)
    pink = (239, 94, 147)
    size = 29 * 8
    panels = [
        ("01  /  Raw XCT", []),
        ("02  /  NIST reference", [(first, sea), (second, gold)]),
        ("03  /  Candidate C", [(join, pink)]),
        ("04  /  Candidate D", [(d1, sea), (d2, gold)]),
    ]
    canvas = Image.new("RGB", (2 * (size + 28), 2 * (size + 41) + 5), "#172322")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for j, (name, masks) in enumerate(panels):
        rgb = raw.copy()
        offx = (j % 2) * (size + 28) + 14
        offy = (j // 2) * (size + 41) + 8
        draw.text((offx, offy), name, font=font, fill="#f0f4e9")
        for mask, col in masks:
            sel = mask[z][box]
            rgb[sel] = np.rint(0.32 * rgb[sel] + 0.68 * np.array(col)).astype("u1")
        canvas.paste(
            Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST), (offx, offy + 25)
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(
        f"set1sample3raw_{src_z:04d}.tif; X [{x},{x + 29}); Y [{y},{y + 29}); C shared component visible in 2D: {rank(z)[0]}"
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("candidate_c", type=Path)
    a.add_argument("candidate_d", type=Path)
    a.add_argument("out", type=Path)
    s = a.parse_args()
    render(s.candidate_c, s.candidate_d, s.out)
