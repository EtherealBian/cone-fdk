"""Render the qualified noisy case: discrete decision map and the selected ray's nine losses."""

import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw(job, cell, path):
    doc = json.loads((job / "manifest.json").read_bytes())
    h, w = doc["shape_ha"]
    levels = doc["levels_per_mm"]
    labels = np.fromfile(job / "class.u8", "u1").reshape(h, w)
    rr, cc = np.where((labels == 1) | (labels == 2))
    x0 = max(0, min(cc) - 3)
    x1 = min(w, max(cc) + 4)
    y0 = max(0, min(rr) - 3)
    y1 = min(h, max(rr) + 4)
    ink = "#123a3c"
    muted = "#465e61"
    bg = "#faf8f2"
    colors = ["#d9e3e0", "#edaa45", "#17615f", "#6a6473"]
    page = Image.new("RGB", (1060, 492), bg)
    pen = ImageDraw.Draw(page)

    def text(x, y, value, size, color=ink, anchor="lt"):
        pen.text((x, y), value, font=ImageFont.load_default(size=size), fill=color, anchor=anchor)

    def box(x, y, dx, dy, color):
        pen.rectangle((x, y, x + dx - 1, y + dy - 1), fill=color)

    text(32, 19, "CONEFDK   /   WALL MEASUREMENT NOTE", 14, muted)
    text(32, 49, "Where does the loss decision change?", 32)
    text(
        32,
        96,
        "Noisy simulated remount  /  Three edge levels per scan  /  Loss gate: 0.30 mm (inclusive)",
        17,
        muted,
    )
    pen.line((32, 128, 1028, 128), fill="#b5c5c0")
    text(34, 147, "Material grid / crop around the relief", 20)
    x, y = 73, 185
    step = min(15, 465 // (x1 - x0), 222 // (y1 - y0))
    for r in range(y0, y1):
        for c in range(x0, x1):
            box(x + (c - x0) * step, y + (y1 - 1 - r) * step, step, step, colors[labels[r, c]])
    text(x - 8, y, str(y1 - 1), 14, muted, "rt")
    text(x - 8, y + (y1 - y0) * step, str(y0), 14, muted, "rb")
    text(x, y + (y1 - y0) * step + 5, str(x0), 14, muted)
    text(x + (x1 - x0) * step, y + (y1 - y0) * step + 5, str(x1 - 1), 14, muted, "rt")
    text(x + (x1 - x0) * step // 2, y + (y1 - y0) * step + 5, "material column", 14, muted, "mt")
    a = x + (cell["column"] - x0) * step
    b = y + (y1 - 1 - cell["row"]) * step
    pen.rectangle((a, b, a + step - 1, b + step - 1), outline="#171c1b", width=2)
    text(558, 147, f"Outlined cell / row {cell['row']}, column {cell['column']}", 20)
    text(558, 180, "Radial loss (mm). Columns: inspected level (/mm)", 15, muted)
    x, y = 684, 233
    bw, bh = 104, 47
    for i, v in enumerate(levels):
        text(x + (i + 0.5) * bw, y - 6, f"{v:.3f}", 16, muted, "mb")
        text(x - 12, y + (i + 0.5) * bh, f"{v:.3f}", 16, muted, "rm")
    text(558, 222, "Reference", 14, muted)
    text(558, 240, "level (/mm)", 14, muted)
    for i, p in enumerate(cell["pairs"]):
        r, c = divmod(i, 3)
        lx = x + c * bw
        ly = y + r * bh
        box(lx, ly, bw - 2, bh - 2, colors[2 if p["flagged"] else 0])
        text(
            lx + bw / 2,
            ly + bh / 2,
            f"{p['loss_mm']:.4f}",
            19,
            "white" if p["flagged"] else ink,
            "mm",
        )
    text(
        558,
        391,
        f"{sum(p['flagged'] for p in cell['pairs'][::4])}/3 same-level pairs flag this cell; {sum(p['flagged'] for p in cell['pairs'])}/9 independent pairs do.",
        14,
        muted,
    )
    pen.line((32, 432, 1028, 432), fill="#b5c5c0")
    x = 34
    for k, title, dx in [
        (0, "Always below", 164),
        (1, "Threshold-sensitive", 207),
        (2, "Always flagged", 172),
        (3, "Unknown", 128),
    ]:
        box(x, 450, 14, 14, colors[k])
        text(x + 21, 450, title, 15)
        x += dx
    text(1028, 450, "Map: all nine combinations", 15, muted, "rt")
    page.save(path)
