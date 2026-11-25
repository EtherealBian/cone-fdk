"""Render the gauge plate from a qualification JSON. No design dimensions used for the sketch."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw(report, output):
    s = json.loads(report.read_bytes())["settings"]
    qualified = s["detector-2x_radial-2x"]
    row = qualified["cases"]["remounted"]["middle_row"]["measurement"]
    bg, ink, muted, teal, red, rule = (
        "#f7f6ef",
        "#192d32",
        "#516267",
        "#087e78",
        "#ab482a",
        "#c7d0c8",
    )
    im = Image.new("RGB", (1120, 552), bg)
    d = ImageDraw.Draw(im)

    def text(x, y, string, size=16, color=ink, anchor="la"):
        d.text((x, y), string, font=ImageFont.load_default(size=size), fill=color, anchor=anchor)

    text(30, 18, "CONEFDK   /   MEASUREMENT NOTE  02", 14, teal)
    text(28, 42, "Does the measured diameter change after a remount?", 34)
    text(
        30,
        93,
        "Three simulated tube designs   /   360 views per scan   /   fixed threshold   /   RTX 5060 Laptop",
        17,
        muted,
    )
    d.line((30, 128, 1090, 128), fill=rule, width=2)
    d.line((455, 146, 455, 470), fill=rule, width=2)
    text(30, 148, "01   Fitted section", 20)
    text(30, 181, "Remounted tube at z = 0 mm", 15, muted)

    def point(xy):
        return tuple(np.array([183.0, 332.0]) + np.array([10.5, -10.5]) * xy)

    def xy(edge):
        return np.array(row[edge]["center_xyz_mm"][:2])

    for edge, col in [("outer", "#c3d8cf"), ("inner", bg)]:
        x, y = point(xy(edge))
        r = row[edge]["diameter_mm"] * 5.25
        d.ellipse((x - r, y - r, x + r, y + r), fill=col, outline=ink, width=2)
    for edge in ["outer", "inner"]:
        x, y = point(xy(edge))
        d.line((x - 4, y, x + 4, y), fill=ink)
        d.line((x, y - 4, x, y + 4), fill=ink)
    direction = xy("inner") - xy("outer")
    direction /= np.linalg.norm(direction)
    ends = [point(xy(e) + direction * row[e]["diameter_mm"] / 2) for e in ["inner", "outer"]]
    d.line(ends, fill=red, width=4)
    x, y = ends[-1]
    d.line([(x, y), (x + 18, y - 12), (x + 37, y - 12)], fill=red, width=1)
    text(x + 40, y - 27, "Model gap", 14, red)
    text(x + 40, y - 9, f"{row['model_min_gap_mm']:.3f} mm", 18, ink)
    text(
        30,
        447,
        f"Bore / OD: {row['inner']['diameter_mm']:.3f} / {row['outer']['diameter_mm']:.3f} mm",
        16,
        muted,
    )
    text(480, 148, "02   Outer diameter change after simulated remount", 20)
    text(480, 181, "Maximum over 17 paired heights (micrometres)", 15, muted)
    text(480, 223, "Detector / radial pitch (mm)", 14, muted)
    x0, scale = 725, 16.0
    gate = 1000 * json.loads(report.read_bytes())["limits"]["paired_delta_mm"]
    gx = x0 + gate * scale
    d.line((gx, 251, gx, 428), fill=ink, width=1)
    for j, v in enumerate(s.values()):
        y = 262 + 43 * j
        change = 1000 * v["max_paired_changes_mm"]["remounted"]["outer_diameter"]
        end = x0 + scale * change
        col = teal if change <= gate else red
        text(480, y, f"{v['detector_pitch_mm']:.2f}    /    {v['radial_pitch_mm']:.3f}", 17)
        d.rectangle((x0, y + 4, end, y + 15), fill=col)
        text(end + 8, y - 2, f"{change:.2f}", 16, col)
    text(gx, 432, f"{gate:g} micrometre gate", 14, ink, "ma")
    d.rectangle((0, 490, 1120, 552), fill=ink)
    text(
        30,
        508,
        "Simulation only. A finer detector is a new acquisition; it cannot recover information from existing scans.",
        17,
        "#d1dfd6",
    )
    im.save(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    draw(a.report, a.output)
