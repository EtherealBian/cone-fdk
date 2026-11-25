"""Plot per-plane maxima from a fully verified snapshot. No smoothing or subsampling."""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_physical_snapshot as audit  # noqa: E402


def svg(values, case):
    audit.need(
        case in {"walnut720", "pine90", "pine720"}
        and len(values) >= 2
        and all(math.isfinite(v) and v >= 0 for v in values),
        "invalid profile",
    )
    n = len(values)
    z = max(range(n), key=values.__getitem__)
    peak = values[z]
    top = peak or 1

    def x(i):
        return 54 + 450 * i / (n - 1)

    def y(v):
        return 130 - 94 * v / top

    points = " ".join(f"{x(i):.2f},{y(v):.2f}" for i, v in enumerate(values))
    cx = x(z)
    cy = y(peak)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 540 168" role="img" aria-labelledby="title desc">
<title id="title">Peak absolute difference by axial plane: {case}</title>
<desc id="desc">One point per reported plane, z=0 through {n - 1}. First maximum {peak:.9g} at z={z}. Linear axes; reconstructed intensity, not distance.</desc>
<rect width="540" height="168" rx="8" fill="#f7f8f4"/>
<g fill="#263e3b" font-family="system-ui,sans-serif" font-size="12"><text x="14" y="19">max |native − ASTRA|</text><text x="517" y="19" text-anchor="end">first peak · z = {z}</text>
<path d="M54 36H504 M54 83H504 M54 130H504" fill="none" stroke="#ccd6d1" stroke-dasharray="3 4"/>
<text x="50" y="40" text-anchor="end" font-size="10">{top:.2g}</text><text x="50" y="133" text-anchor="end">0</text>
<polyline points="{points}" stroke="#07776d" stroke-width="1.5" stroke-linejoin="round" fill="none"/>
<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4.5" fill="#a33a20" stroke="#fff" stroke-width="1.4"/>
<text x="54" y="154">0</text><text x="279" y="154" text-anchor="middle">axial plane z</text><text x="504" y="154" text-anchor="end">{n - 1}</text></g></svg>\n'''


def draw(folder, pin, case):
    profiles = {}
    audit.review(folder, pin, profiles=profiles)
    audit.need(case in profiles, "case not submitted")
    return svg(profiles[case], case)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("catalog_sha256")
    p.add_argument("case")
    p.add_argument("new_svg", type=Path)
    a = p.parse_args()
    try:
        out = draw(a.folder, a.catalog_sha256, a.case).encode("utf-8")
        with a.new_svg.open("xb") as f:
            f.write(out)
    except (
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        OSError,
        OverflowError,
        RecursionError,
    ) as e:
        p.exit(1, f"plot refused: {e}\n")


if __name__ == "__main__":
    main()
