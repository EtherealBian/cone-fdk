"""Plot actual outer-surface residuals on the middle row of the packaged review example."""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from axis_probe import RECORD, angle
from PIL import Image, ImageDraw, ImageFont
from validate_bore import cpu_circle


def draw(pack, path):
    curves = {}
    with zipfile.ZipFile(pack) as f:
        fit = json.loads(f.read("axis.json"))
        w, h = fit["columns"], fit["rows"]
        j = h // 2
        gate = json.loads(f.read("sections.json"))["max_residual_mm"] * 1000
        rough = np.array(f.read("coarse.grid").split()[7:10], dtype=float)
        tilt = angle(rough, fit["measurement"]["axis_unit"])
        phi = 2 * np.pi * np.arange(w) / w
        rays = np.c_[np.cos(phi), np.sin(phi)]
        for key in ["coarse", "aligned"]:
            with f.open(key + ".wall") as file:
                file.seek(j * w * RECORD.itemsize)
                row = np.frombuffer(file.read(w * RECORD.itemsize), RECORD)
            if row["status"].any():
                raise ValueError("incomplete illustrative section")
            points = row["outer"][:, None] * rays
            q, _, _ = cpu_circle(points)
            curves[key] = (np.linalg.norm(points - q[:2], axis=1) - q[2]) * 1000
    dark = "#21332f"
    muted = "#50645f"
    orange = "#ad4b29"
    teal = "#096e68"
    line = "#c8d3c8"
    paper = "#f7f7ef"
    im = Image.new("RGB", (1040, 530), paper)
    d = ImageDraw.Draw(im)

    def text(x, y, v, size, color=dark):
        d.text((x, y), v, font=ImageFont.load_default(size=size), fill=color)

    text(40, 19, "CONEFDK   /   MEASUREMENT NOTE  04", 15, teal)
    text(38, 46, f"Tube tilted {tilt:.1f} degrees: two ways to cut the same scan", 32)
    text(
        40,
        96,
        "Outer-circle residual on the middle row. Same detector data, two reconstruction planes.",
        18,
        muted,
    )
    left, right, top, bottom = 88, 978, 168, 395
    center = (top + bottom) / 2
    peak = max(max(abs(v)) for v in curves.values())
    scale = 25 * np.ceil(max(peak, gate) / 25)

    def y(v):
        return center - v * (bottom - top) / (2 * scale * 1.15)

    d.rectangle((left, y(gate), right, y(-gate)), fill="#dceadf")
    for v in [-scale, -gate, 0, gate, scale]:
        yy = y(v)
        d.line((left, yy, right, yy), fill=line)
        text(46, yy - 8, f"{v:+.0f}", 13, muted)
    text(40, 137, "Residual (um)", 15, muted)
    for deg in range(0, 361, 60):
        x = left + (right - left) * deg / 360
        d.line((x, top, x, bottom), fill=line)
        text(x - 10, bottom + 6, str(deg), 13, muted)
    for key, color in [("coarse", orange), ("aligned", teal)]:
        d.line(
            [(left + (right - left) * i / w, y(v)) for i, v in enumerate(curves[key])],
            fill=color,
            width=3,
        )
    t = 431
    for k, (key, title, color) in enumerate(
        [("coarse", "Scanner plane", orange), ("aligned", "Axis-normal plane", teal)]
    ):
        x = 60 + 365 * k
        d.line((x, t + 12, x + 30, t + 12), fill=color, width=3)
        text(x + 40, t, title, 17, color)
        text(x + 40, t + 23, f"Peak {max(abs(curves[key])):.1f} um", 16)
    text(763, t, f"Shading: +/-{gate:g} um", 16, muted)
    text(763, t + 23, "Azimuth in degrees", 15, muted)
    text(
        40,
        499,
        f"Simulated finite tube  /  {w} samples per curve  /  azimuth is local to each sampling plane, not a material landmark.",
        15,
        muted,
    )
    im.save(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pack", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    draw(a.pack, a.output)
