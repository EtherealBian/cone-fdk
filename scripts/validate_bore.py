"""Projection-to-gauge qualification: re-unwrapping vs independently projected simulated remount."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from bore import gauge
from common import BUILD, REPORT, reconstruct
from compare_native import sha
from validate_cylinder import spec
from validate_roi import require
from validate_stack import require_accounting, save_json
from validate_wall import CASES, DIMS, MU, STEPS, project
from wall import WALL_DTYPE, measure

RESIDUAL = 0.025  # mm; applies to the worst sampled point, not the averaged fit
LIMITS = {
    "diameter_mm": 0.025,
    "world_center_mm": 0.012,
    "eccentricity_mm": 0.012,
    "model_min_gap_mm": 0.016,
    "cpu_fit_mm": 1e-8,
    "paired_delta_mm": 0.012,
}


def cpu_circle(points):
    """Independent dense QR/SVD least squares on all point residuals with three unknowns."""
    center = points.mean(axis=0)
    q = np.r_[center, np.linalg.norm(points - center, axis=1).mean()]
    for _ in range(30):
        dv = q[:2] - points
        dist = np.linalg.norm(dv, axis=1)
        jac = np.c_[dv / dist[:, None], -np.ones(len(dist))]
        step = np.linalg.lstsq(jac, q[2] - dist, rcond=None)[0]
        q += step
        if np.linalg.norm(step) < 1e-13:
            break
    else:
        raise ValueError("CPU circle did not converge")
    residual = np.linalg.norm(points - q[:2], axis=1) - q[2]
    return q, np.sqrt(np.mean(residual**2)), np.max(abs(residual))


def direction(deg):
    angle = np.deg2rad(deg)
    return np.array([np.cos(angle), np.sin(angle)])


# Each scan entry is a separate integration of its own finite solid geometry.
SCANS = {
    "nominal": (CASES["near-seam"], (0, 0), None),
    "remounted": ((9.6, 6.2, 1.4, 61), (0.65, -0.45), None),
    "second-part": (CASES["oblique-bore"], (0, 0), None),
    "concentric": ((9.15, 6.45, 0.0, 0), (-0.45, 0.55), None),
    "ellipse": ((9.6, 6.1, 0.8, 20), (0, 0), 5.6),
}
FRAMES = {
    "nominal": ("nominal", (0, 0), 0),
    "reframed": ("nominal", (0.40, -0.35), 37),
    "remounted": ("remounted", (0.30, -0.15), -23),
    "second-part": ("second-part", (0, 0), 0),
    "concentric": ("concentric", (-0.1, 0.15), 19),
    "ellipse": ("ellipse", (0, 0), 0),
}


def evaluate(base, scans_dir, radial_factor):
    base.mkdir(exist_ok=True, parents=True)
    w, h, d = DIMS
    dims = (w, h, 1 + (d - 1) * radial_factor)
    steps = (STEPS[0], STEPS[1] / radial_factor, STEPS[2])
    summary, series = {}, {}
    for name, (scan, origin, deg) in FRAMES.items():
        part, mount, ey = SCANS[scan]
        seam = direction(deg)
        grid = spec(base / (name + ".grid"), dims, (*origin, 0), (0, 0, 1), (*seam, 0), steps)
        atlas, wall, out = (base / (name + ext) for ext in (".atlas", ".wall", ".sections.json"))
        rec = reconstruct(scans_dir / scan, atlas, 512, 26, 32, 16, cylinder=grid)
        require_accounting(rec, scans_dir / scan, atlas)
        m = measure(atlas, grid, wall, MU / 2)
        require(
            m["status_counts"]["measured"] == w * h, "fixture must yield all single-wall profiles"
        )
        stats = gauge(wall, grid, out, RESIDUAL)
        rows = json.loads(out.read_bytes())["rows"]
        require(
            len(rows) == h
            and stats["redzones_verified"]
            and stats["peak_owned_bytes"] <= stats["budget_bytes"]
            and stats["input_bytes"] == wall.stat().st_size
            and stats["output_bytes"] == out.stat().st_size,
            "bore accounting",
        )
        summary[name] = dict(
            projections_sha256=sha((scans_dir / scan).with_suffix(".f32")),
            grid=grid.read_text().splitlines(),
            wall_sha256=sha(wall),
            sections_sha256=sha(out),
            native=stats,
            peak_inner_residual_mm=max(r["residual_mm"]["inner_peak"] for r in rows),
            middle_row=rows[h // 2],
        )
        if ey:
            require(
                stats["rejected"] == h
                and all(
                    r["status"] == "noncircular"
                    and r["measurement"] is None
                    and r["residual_mm"]["inner_peak"] > RESIDUAL
                    for r in rows
                ),
                "elliptical bore should be rejected",
            )
            summary[name]["inner_semiaxes_mm"] = [part[1], ey]
            continue
        require(stats["measured"] == h, f"{name}: rejected circular section")
        data = np.fromfile(wall, WALL_DTYPE).reshape(h, w)
        ro, ri, e, phase = part
        true_outer = np.array(mount)
        true_inner = true_outer + e * direction(phase)
        design = np.array([2 * ri, 2 * ro, e, ro - ri - e])
        vals = []
        centers = []
        cpu_error = 0.0
        angles = 2 * np.pi * np.arange(w) / w + np.deg2rad(deg)
        unit = np.c_[np.cos(angles), np.sin(angles)]
        for j, row in enumerate(rows):
            val = row["measurement"]
            z = (j - (h - 1) / 2) * STEPS[2]
            require(
                row["row"] == j and row["axial_offset_mm"] == z and row["valid_columns"] == w,
                "row contract",
            )
            for edge, xy in [("inner", true_inner), ("outer", true_outer)]:
                circle = val[edge]
                ctr = circle["center_xyz_mm"]
                diameter = circle["diameter_mm"]
                centers.append(np.linalg.norm(ctr - np.r_[xy, z]))
                fit, rms, peak = cpu_circle(origin + data[edge][j, :, None] * unit)
                cpu_error = max(
                    cpu_error,
                    np.linalg.norm(fit[:2] - ctr[:2]),
                    abs(2 * fit[2] - diameter),
                    abs(rms - row["residual_mm"][edge + "_rms"]),
                    abs(peak - row["residual_mm"][edge + "_peak"]),
                )
            vals.append(
                [
                    val["inner"]["diameter_mm"],
                    val["outer"]["diameter_mm"],
                    val["eccentricity_mm"],
                    val["model_min_gap_mm"],
                ]
            )
        series[name] = np.array(vals)
        errs = abs(series[name] - design).max(axis=0)
        metrics = dict(
            diameter_mm=float(errs[:2].max()),
            world_center_mm=float(max(centers)),
            eccentricity_mm=float(errs[2]),
            model_min_gap_mm=float(errs[3]),
            cpu_fit_mm=float(cpu_error),
        )
        require(all(v <= LIMITS[k] for k, v in metrics.items()), f"{name}: {metrics}")
        summary[name].update(
            design=dict(
                inner_diameter_mm=2 * ri,
                outer_diameter_mm=2 * ro,
                outer_xy_mm=list(mount),
                inner_xy_mm=true_inner.tolist(),
                eccentricity_mm=e,
                model_min_gap_mm=ro - ri - e,
            ),
            max_errors=metrics,
        )
        print(name, metrics, flush=True)
    require(
        summary["reframed"]["projections_sha256"]
        == summary["nominal"]["projections_sha256"]
        != summary["remounted"]["projections_sha256"]
        and summary["reframed"]["wall_sha256"] != summary["nominal"]["wall_sha256"],
        "no actual pose challenge",
    )
    paired = {
        key: dict(
            zip(
                ("inner_diameter", "outer_diameter", "eccentricity", "model_min_gap"),
                abs(series[key] - series["nominal"]).max(axis=0).tolist(),
                strict=True,
            )
        )
        for key in ("reframed", "remounted")
    }
    return dict(
        cases=summary,
        max_paired_changes_mm=paired,
        paired_gate_passed=all(
            max(v.values()) <= LIMITS["paired_delta_mm"] for v in paired.values()
        ),
    )


def qualify(base, reports):
    reports.mkdir(exist_ok=True, parents=True)
    settings = {}
    for df in (1, 2):
        scans_dir = base / f"detector-{df}x"
        scans_dir.mkdir(exist_ok=True, parents=True)
        for scan, (part, mount, ey) in SCANS.items():
            project(scans_dir / scan, *part, mount_xy=mount, bore_y_radius=ey, detector_factor=df)
        for rf in (1, 2):
            key = f"detector-{df}x_radial-{rf}x"
            result = evaluate(base / key, scans_dir, rf)
            settings[key] = dict(
                detector_pitch_mm=0.14 / df, radial_pitch_mm=STEPS[1] / rf, **result
            )
    require(
        settings["detector-2x_radial-2x"]["paired_gate_passed"],
        "qualified setting failed paired repeatability",
    )
    save_json(
        reports / "bore-gauge.json",
        dict(
            protocol="Noiseless analytic finite-tube projections. In-plane rigid simulated remount, not a physical repeat scan. All four sampling settings retained.",
            binary_sha256={n: sha(BUILD / n) for n in ("cone-fdk.exe", "kernels.ptx")},
            limits=LIMITS,
            max_residual_mm=RESIDUAL,
            threshold=MU / 2,
            settings=settings,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path, default=REPORT)
    ap.add_argument("--keep", type=Path)
    a = ap.parse_args()
    with TemporaryDirectory() as t:
        qualify(a.keep or Path(t), a.reports)
