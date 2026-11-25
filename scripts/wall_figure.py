"""Static qualification plate drawn from measured FP64 wall records (Pillow only)."""

import math

from PIL import Image, ImageDraw, ImageFont


def draw_figure(path, results, series, negative):
    bg, ink, muted, line = "#f7f6ef", "#192d32", "#516267", "#c7d0c8"
    teal, red = "#087e78", "#ab482a"
    im = Image.new("RGB", (1120, 654), bg)
    d = ImageDraw.Draw(im)

    def text(x, y, s, size=16, color=ink, anchor="la"):
        d.text((x, y), s, fill=color, font=ImageFont.load_default(size=size), anchor=anchor)

    def poly(vals, x, y, w, h, lo, hi):
        return [
            (x + i * w / len(vals), y + h - (min(hi, max(lo, v)) - lo) * h / (hi - lo))
            for i, v in enumerate([*vals, vals[0]])
        ]

    text(32, 18, "CONEFDK   /   MEASUREMENT NOTE  01", 14, teal)
    text(30, 42, "Where is the thin side of an eccentric tube?", 34)
    text(
        32,
        92,
        "Synthetic CT   /   360 views per part   /   17 axial rows   /   1-degree azimuth samples",
        17,
        muted,
    )
    for n, (name, r) in enumerate(results.items()):
        top = 132 + 215 * n
        ss = series[name]
        m = r["native_wall"]
        deg = r["truth_thinnest_deg"]
        a = math.radians(deg)
        d.line((32, top, 1088, top), fill=line, width=2)
        text(32, top + 12, f"0{n + 1}   {name.upper()}", 17, teal)
        x, y, R = 109, top + 115, 63

        def circ(x, y, R, fill):
            d.ellipse((x - R, y - R, x + R, y + R), fill=fill)

        circ(x, y, R, "#c3d8cf")
        e = R * r["eccentricity_mm"] / r["outer_mm"]
        rad = R * r["bore_mm"] / r["outer_mm"]
        circ(x + e * math.cos(a), y - e * math.sin(a), rad, bg)
        d.line((x - 4, y, x + 4, y), fill=ink)
        d.line((x, y - 4, x, y + 4), fill=ink)
        d.arc(
            (x - R - 3, y - R - 3, x + R + 3, y + R + 3),
            -deg - 7.5,
            -deg + 7.5,
            fill=red,
            width=5,
        )
        text(x, top + 190, "Design section", 14, muted, "ma")
        text(207, top + 49, "Thin sector center", 14, muted)
        text(206, top + 72, f"{m['thinnest_complete_sector']['azimuth_deg']:.0f} deg", 28)
        text(207, top + 120, "Minimum sampled", 14, muted)
        text(206, top + 144, f"{m['min_valid_point']['mm']:.3f} mm", 24)
        for graph in range(2):
            x = 397 if graph == 0 else 777
            y = top + 66
            w = 283 if graph == 0 else 268
            h = 93
            if graph == 0:
                lo, hi = 1.5, 5.0
                title = "Radial thickness (mm)"
                legend = "Design / measured axial mean"
            else:
                lo, hi = -16, 16
                title = "Thickness error (micrometres)"
                legend = "Signed range across all 17 rows"
            text(x, top + 18, title, 18)
            text(x, top + 42, legend, 13, muted)
            for v in (lo, (lo + hi) / 2, hi):
                yy = y + h - (v - lo) * h / (hi - lo)
                d.line((x, yy, x + w, yy), fill=line, width=1)
                text(x - 6, yy, f"{v:g}", 12, muted, "rm")
            for a_deg in (0, 90, 180, 270, 360):
                xx = x + a_deg * w / 360
                d.line((xx, y + h, xx, y + h + 4), fill=muted)
                text(xx, y + h + 7, str(a_deg), 12, muted, "ma")
            if graph == 0:
                d.line(poly(ss["truth"], x, y, w, h, lo, hi), fill=ink, width=4)
                d.line(poly(ss["mean"], x, y, w, h, lo, hi), fill=teal, width=2)
            else:
                upper = poly(ss["max_error_um"], x, y, w, h, lo, hi)
                lower = poly(ss["min_error_um"], x, y, w, h, lo, hi)
                d.polygon([*upper, *lower[::-1]], fill="#e8c4ad")
                d.line(upper, fill=red, width=1)
                d.line(lower, fill=red, width=1)
            text(x + w / 2, top + 187, "Azimuth from seam (deg)", 13, muted, "ma")
    d.rectangle((0, 574, 1120, 654), fill=ink)
    refused = negative["native_wall"]["status_counts"]["multiple"]
    text(
        32,
        587,
        f"Negative control   /   Lined tube: {refused:,} of {refused:,} profiles refused as multiple walls.",
        19,
        "#f6f2e8",
    )
    text(
        32,
        620,
        "These are radial distances on simulated parts, not shortest 3D distances or certified real-part tolerances.",
        15,
        "#c4d2cd",
    )
    im.save(path)
