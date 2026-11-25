"""Plot measured decreases on four maps, with a common scale and explicit unknowns."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw(path, maps, control, results):
    page = Image.new("RGB", (1040, 650), "#f6f4ed")
    pen = ImageDraw.Draw(page)

    def text(x, y, label, size, color="#18343a"):
        pen.text((x, y), label, fill=color, font=ImageFont.load_default(size=size))

    text(38, 20, "CONEFDK   /   MEASUREMENT NOTE  05", 15, "#187476")
    text(36, 45, "Which wall changed after the tube was remounted?", 34)
    text(
        38,
        97,
        "Same tube, two scanner poses. Each map is radial thickness decrease in mm. Synthetic projections.",
        17,
    )
    palette = np.array([[28, 54, 64], [30, 88, 104], [212, 158, 76], [219, 78, 41]])
    cases = [
        (
            "scanner-only",
            "Unchanged / fixed scanner grid",
            f"{control['coverage']['flagged']:,} falsely flagged cells",
        ),
        (
            "repeat",
            "Unchanged / fixture registered",
            f"{results['repeat']['coverage']['flagged']} flagged cells",
        ),
        ("deepened", "Old outside flat / deepened", "One region across the 0 / 360 seam"),
        (
            "changed",
            "New inside relief",
            f"One region, peak at {results['changed']['peak_azimuth_deg']:g} degrees",
        ),
    ]
    for i, (name, title, subtitle) in enumerate(cases):
        x = 38 + (i % 2) * 508
        y = 145 + (i // 2) * 207
        d = maps[name]
        valid = np.isfinite(d)
        v = np.clip(np.nan_to_num(d), 0, 0.8)
        rgb = np.stack(
            [np.interp(v, [0, 0.30, 0.50, 0.8], palette[:, k]) for k in range(3)], axis=-1
        ).astype("u1")
        rgb[~valid] = [171, 172, 165]
        text(x, y, title, 21)
        page.paste(
            Image.fromarray(rgb[::-1]).resize((450, 96), Image.Resampling.NEAREST), (x, y + 36)
        )
        text(x, y + 138, subtitle, 16, "#495c60")
        text(x, y + 161, "Scanner azimuth" if i == 0 else "Material azimuth", 13)
        text(x + 286, y + 161, "0 to 360 deg / z: +/-4 mm", 13)
    pen.line((38, 553, 1000, 553), fill="#c4cfca", width=1)
    text(38, 573, "Loss scale", 16)
    for i, v in enumerate([0, 0.30, 0.50, 0.80]):
        x = 154 + i * 125
        pen.rectangle((x, 576, x + 21, 591), fill=tuple(palette[i]))
        text(x + 28, 573, f"{v:.2f} mm", 15)
    pen.rectangle((674, 576, 695, 591), fill=(171, 172, 165))
    text(703, 573, "Unmeasured", 15)
    text(
        38,
        615,
        "Flag threshold: 0.30 mm. Negative changes are shown at zero. Grey is not a zero-thickness measurement.",
        15,
        "#495c60",
    )
    page.save(path)
