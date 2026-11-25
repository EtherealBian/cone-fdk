"""Render the numerical loss map and axial footprint. Display does not modify detection."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw(path, loss, doc, truth):
    paper = "#f7f6ef"
    ink = "#182f37"
    muted = "#516267"
    warm = "#b4472b"
    blue = "#166d7a"
    img = Image.new("RGB", (1040, 600), paper)
    d = ImageDraw.Draw(img)

    def text(x, y, s, size=16, color=ink):
        d.text((x, y), s, fill=color, font=ImageFont.load_default(size=size))

    text(30, 16, "CONEFDK   /   MEASUREMENT NOTE  03", 15, blue)
    text(29, 43, "Where did the wall lose material?", 36)
    text(
        31,
        94,
        "Reference minus inspected radial thickness   /   simulated, registered tube pair",
        18,
        muted,
    )
    x0, y0, mw, mh = 67, 156, 900, 205
    text(31, 127, "A", 17, blue)
    text(66, 127, "Unwrapped wall   /   white contour = design footprint at 0.30 mm", 16)
    v = np.clip(loss / 0.85, 0, 1)
    colors = np.array([[21, 44, 53], [29, 94, 111], [226, 159, 81], [229, 87, 48]])
    rgb = np.stack(
        [np.interp(v, [0, 0.23, 0.60, 1], colors[:, k]) for k in range(3)], axis=-1
    ).astype("u1")
    mask = (truth[0] + truth[1]) >= doc["threshold_mm"]
    inside = mask.copy()
    for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
        inside &= np.roll(mask, shift, axis=axis)
    rgb[mask & ~inside] = [247, 245, 229]
    img.paste(Image.fromarray(rgb[::-1]).resize((mw, mh), Image.Resampling.NEAREST), (x0, y0))
    for i in range(5):
        text(x0 + i * mw / 4 - 6, y0 + mh + 5, f"{90 * i} deg", 14, muted)
    for i, z in [(0, "+4"), (0.5, "0"), (1, "-4")]:
        text(30, y0 + i * (mh - 16), z, 15, muted)
    text(28, 237, "z", 13, muted)
    text(30, 401, "B", 17, blue)
    text(66, 401, "Two connected regions in the inspected map", 18)
    for reg, label, color, x in zip(
        doc["regions"],
        ["Outside flat", "Inside bore relief"],
        [warm, blue],
        [66, 524],
        strict=True,
    ):
        d.line((x, 439, x + 411, 439), fill=color, width=3)
        text(x, 451, label, 22, color)
        text(x, 486, f"Peak loss   {reg['peak']['loss_mm']:.3f} mm", 20)
        text(
            x,
            518,
            f"Footprint   {reg['area_mm2']:.2f} mm2     /     {reg['cells']} cells",
            16,
            muted,
        )
    text(
        31,
        565,
        "Area on the 9.4 mm reference cylinder. Periodic azimuth; bounded axial range. No physical part scanned.",
        15,
        muted,
    )
    img.save(path)
