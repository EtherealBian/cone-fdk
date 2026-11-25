"""Measure three known circular cuts through the analytic phantom's spherical inclusion."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, DATA, REPORT, reconstruct
from compare_native import sha
from phantom import SPHERE
from validate_roi import require


def run(scan, dest):
    pinned = json.loads((REPORT / "native-ab.json").read_text())["cases"]["phantom"]
    inputs = {e: sha(scan.with_suffix(e)) for e in (".geom", ".f32")}
    require(
        inputs == {".geom": pinned["geom_sha256"], ".f32": pinned["projections_sha256"]},
        "unexpected analytic phantom inputs",
    )
    origin = np.array(SPHERE[:3])
    radius = SPHERE[3]
    w, h, du, dv = 97, 83, 0.05, 0.075
    results = []
    # Halfway between the known background and inclusion attenuation; fixed before the runs.
    threshold = 0.04 + SPHERE[4] / 2
    for u, v, d in [
        ([0.8, 0.6, 0], [-0.36, 0.48, 0.8], 0.0),
        ([0, -0.8, 0.6], [1, 0, 0], 0.6),
        ([0.6, 0, -0.8], [0.48, 0.8, 0.36], -0.85),
    ]:
        u, v = np.array(u), np.array(v)
        circle = origin + d * np.cross(u, v)
        center = circle - 0.35 * u + 0.225 * v  # Avoid placing the object at the image center.
        col, row = du * u, dv * v
        expected = 2 * np.sqrt(radius**2 - d**2)
        with TemporaryDirectory() as temp:
            temp = Path(temp)
            spec = temp / "sphere.txt"
            out = temp / "cut.f32"
            spec.write_text(
                "\n".join(
                    [f"CFDKPLANE1 {w} {h}", *(" ".join(map(str, x)) for x in (center, col, row))]
                )
                + "\n",
                encoding="ascii",
            )
            m = reconstruct(scan, out, 512, 32, 32, 16, plane=spec)
            frame = np.fromfile(out, "<f4").reshape(h, w)
            jj, ii = np.nonzero(frame > threshold)
            require(
                len(ii) and min(ii) > 0 and max(ii) < w - 1 and min(jj) > 0 and max(jj) < h - 1,
                "empty or clipped inclusion",
            )
            observed_center = (
                center + (ii.mean() - (w - 1) / 2) * col + (jj.mean() - (h - 1) / 2) * row
            )
            observed = 2 * np.sqrt(len(ii) * du * dv / np.pi)
            error = abs(observed - expected)
            center_error = np.linalg.norm(observed_center - circle)
            require(
                error < 0.10 and center_error < 0.06,
                f"analytic disk outside limits: diameter {error} mm, center {center_error} mm",
            )
            require(
                m["redzones_verified"]
                and m["peak_owned_bytes"] <= (32 << 20)
                and m["output_bytes"] == 4 * w * h,
                "memory or shape gate",
            )
            results.append(
                dict(
                    signed_normal_offset_mm=d,
                    manifest=spec.read_text().splitlines(),
                    sha256=sha(out),
                    mask_pixels=len(ii),
                    expected_diameter_mm=expected,
                    observed_area_equivalent_diameter_mm=observed,
                    diameter_abs_error_mm=error,
                    expected_center_xyz_mm=circle.tolist(),
                    observed_center_xyz_mm=observed_center.tolist(),
                    center_distance_mm=center_error,
                )
            )
    record = dict(
        dataset="Analytic circular cone projections of a phantom with a spherical inclusion; not a measured part",
        inputs_sha256=inputs,
        binaries_sha256={name: sha(BUILD / name) for name in ("cone-fdk.exe", "kernels.ptx")},
        gpu=m["gpu"],
        sphere_radius_mm=radius,
        threshold_per_mm=threshold,
        pixel_pitch_mm=[du, dv],
        diameter_abs_error_limit_mm=0.10,
        center_distance_limit_mm=0.06,
        method="Threshold every pixel on the cut at the known half contrast. No intensity or geometry fitting. Equivalent diameter = 2 sqrt(mask count * pixel area / pi). Expected diameter = 2 sqrt(radius^2 - signed offset^2).",
        cuts=results,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf8")
    print(
        "Three analytic sphere cuts: worst center / diameter errors (mm):",
        max(r["center_distance_mm"] for r in results),
        max(r["diameter_abs_error_mm"] for r in results),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=DATA / "phantom")
    p.add_argument("--out", type=Path, default=REPORT / "direct-planes-phantom.json")
    a = p.parse_args()
    run(a.scan, a.out)
