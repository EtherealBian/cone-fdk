"""Create a self-contained fixed-window preview of the measured walnut qualification."""

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sweep_frame(image, k, radii):
    # A schematic (not a segment mask) indicates which shell the strip samples.
    w, h = image.size
    frame = Image.new("L", (w + 108, h + 54), 24)
    frame.paste(image, (8, 25))
    pen = ImageDraw.Draw(frame)
    pen.text((8, 5), "CONEFDK  /  WALNUT     |     CYLINDRICAL SWEEP", fill=224)
    pen.text((8, h + 32), "0 deg", fill=210)
    pen.text((w - 60, h + 32), "360 deg / 0", fill=210)
    pen.text((w // 2 - 70, h + 32), "A. Meaney / CC BY 4.0", fill=180)
    x, y = w + 56, 100
    for n, radius in enumerate(radii):
        r = round(37 * radius / radii[-1])
        pen.ellipse(
            (x - r, y - r, x + r, y + r), outline=235 if k == n else 66, width=2 if k == n else 1
        )
    pen.text((w + 21, y + 52), f"r {radii[k]:.1f} mm", fill=240)
    pen.text((w + 21, y + 74), f"{k + 1:02} / {len(radii):02}", fill=170)
    return frame


def write_atlas(a, metrics, reports):
    lo, hi = np.percentile(a, [2, 99])
    frames = []
    animation = []
    radii = metrics["first_radius_mm"] + np.arange(len(a)) * metrics["radial_step_mm"]
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("empty intensity window")
    for k, layer in enumerate(a):
        image = Image.fromarray(
            np.rint(np.clip((layer - lo) / (hi - lo), 0, 1) * 255).astype("u1")
        )
        animation.append(sweep_frame(image, k, radii))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    animation[0].save(
        reports / "cylinder-sweep.gif",
        save_all=True,
        append_images=animation[1:] + animation[-2:0:-1],
        duration=190,
        loop=0,
    )
    fields = (
        "shape_rha",
        "center_mm",
        "axis_unit",
        "seam_unit",
        "tangent_unit",
        "first_radius_mm",
        "radial_step_mm",
        "axial_pitch_mm",
    )
    payload = {k: metrics[k] for k in fields}
    payload.update(frames=frames, window=[lo, hi])
    template = Path(__file__).with_name("cylinder_atlas.html").read_text(encoding="utf8")
    html = template.replace(
        "@@ATLAS@@",
        json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c"),
    )
    (reports / "walnut-cylinder-atlas.html").write_text(html, encoding="utf8", newline="\n")
    return dict(
        window_per_mm=[lo, hi],
        encoding="linear 8-bit grayscale PNG; same 2nd–99th percentile limits over all sheets",
        frames=len(frames),
    )
