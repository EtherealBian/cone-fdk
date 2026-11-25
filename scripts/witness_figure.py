"""Plot the measured scanner-XY windows from the pivot audit, without re-running a fit."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from witness_probe import audit


def draw(pack, path):
    checked = audit(pack)
    page = Image.new("RGB", (920, 420), "#f7f5ee")
    pen = ImageDraw.Draw(page)
    ink = "#183438"
    muted = "#406167"
    observed = "#65e6bc"
    predicted = "#ffad55"

    def text(x, y, s, size, color=ink):
        pen.text((x, y), s, font=ImageFont.load_default(size=size), fill=color)

    text(24, 12, "CONEFDK   /   REMOUNT INSPECTION", 14, muted)
    text(22, 33, "The carrier rotated around insert A", 31)
    text(
        24,
        77,
        "Three windows from the same CUDA scout. Compare the fixture prediction with the material insert.",
        16,
    )
    with ZipFile(pack) as z:
        job = json.loads(z.read("manifest.json"))
        fixture = json.loads(z.read("fixture.json"))
        n = fixture["locator_n"]
        pitch = fixture["locator_fov_mm"] / n
        top, width = 137, 205
        for i, (rec, measure) in enumerate(
            zip(job["scans"]["inspected"]["body_witnesses"], checked, strict=True)
        ):
            left = 25 + i * 299
            x, y, _, w, h, d = rec["parent_roi_xyz_nxyz"]
            vox = np.frombuffer(z.read(f"insert-{i + 1}.f32"), "<f4").reshape(d, h, w)
            img = Image.fromarray((np.clip((vox.max(0) - 0.03) / 0.08, 0, 1) * 255).astype("u1"))
            page.paste(img.resize((width, width), Image.Resampling.NEAREST), (left, top))
            text(left, 109, f"Insert {chr(65 + i)}  /  {measure['status'].upper()}", 17, muted)

            def pixel(key, rec=rec, x=x, y=y, w=w, h=h, left=left):
                pos = np.array(rec[key][:2]) / pitch + (n - 1) / 2 - [x, y] + 0.5
                return tuple((np.array([left, top]) + pos * width / [w, h]).tolist())

            px, py = pixel("predicted_scanner_xyz_mm")
            ox, oy = pixel("observed_scanner_xyz_mm")
            pen.ellipse((ox - 8, oy - 8, ox + 8, oy + 8), outline=observed, width=3)
            pen.line((px - 6, py, px + 6, py), fill=predicted, width=2)
            pen.line((px, py - 6, px, py + 6), fill=predicted, width=2)
            text(left, 347, f"3D residual: {measure['residual_mm']:.4f} mm", 16)
    text(
        24,
        380,
        "Orange +  fixture prediction     Green ring  measured centroid     |     Gate: 0.050 mm per insert",
        16,
    )
    text(
        24,
        400,
        "Synthetic tube. Scanner-XY maximum over Z; fixed display range 0.03-0.11 /mm. Residuals use all three coordinates.",
        13,
        muted,
    )
    page.save(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pack", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    draw(a.pack, a.out)
