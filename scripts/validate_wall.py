"""Qualify radial wall thickness from analytic X-ray projections, not voxelized labels."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, REPORT, Geometry, reconstruct
from compare_native import sha
from validate_cylinder import spec
from validate_roi import require
from validate_stack import require_accounting, save_json
from wall import STATUS, WALL_DTYPE, measure
from wall_figure import draw_figure
from wall_probe import probe

CASES = {"near-seam": (9.6, 6.2, 1.4, 358.0), "oblique-bore": (8.8, 5.8, 0.75, 134.0)}
# Fixed before fitting any part. Detector-center rays are integrated at 3 horizontal subpixels.
MU = 0.04
DIMS = (360, 17, 161)
STEPS = (3.6, 0.05, 0.25)
LIMITS = {
    "max_thickness_abs_mm": 0.04,
    "p95_thickness_abs_mm": 0.025,
    "max_interface_abs_mm": 0.035,
    "sector_direction_deg": 4.0,
    "point_min_abs_mm": 0.035,
    "cpu_abs_mm": 1e-11,
}


def project(scan, outer, inner, ecc, phase, *, liner=False):
    g = Geometry(360, 72, 320, 0.14, 0.24, 120.0, 205.0)
    g.save(scan.with_suffix(".geom"))
    u = (np.arange(g.u) - (g.u - 1) / 2) * g.du
    v = (np.arange(g.v)[:, None] - (g.v - 1) / 2) * g.dv
    e = np.deg2rad(phase)
    cx = ecc * np.cos(e)
    cy = ecc * np.sin(e)
    with scan.with_suffix(".f32").open("wb") as f:
        for th in g.angles():
            s, c = np.sin(th), np.cos(th)
            sx, sy = g.R * s, -g.R * c
            view = np.zeros((g.v, g.u))
            for offset in (-1 / 3, 0, 1 / 3):
                x = u + offset * g.du
                n = np.hypot(g.D, x)
                dx = (-g.D * s + c * x) / n
                dy = (g.D * c + s * x) / n
                slope = v / n
                zcap = 7 / abs(slope)  # finite tube z in [-7,7]; even detector height avoids 0/0
                scale = np.hypot(1, slope)
                solids = [(0, 0, outer, 1), (cx, cy, inner, -1)]
                if liner:
                    solids += [(0, 0, 4.4, 1), (0, 0, 3.9, -1)]
                for ox, oy, r, sign in solids:
                    ax, ay = ox - sx, oy - sy
                    along = ax * dx + ay * dy
                    across = ax * dy - ay * dx
                    half = np.sqrt(np.maximum(0, r * r - across * across))
                    length = np.maximum(
                        0, np.minimum(along + half, zcap) - np.maximum(along - half, 0)
                    )
                    view += sign * (MU / 3) * length * scale
            view.astype("<f4").tofile(f)
    return g


def truth(part):
    ro, ri, e, deg = part
    angle = np.deg2rad(np.arange(DIMS[0]) * 360 / DIMS[0] - deg)
    dot = e * np.cos(angle)
    inside = dot + np.sqrt(ri * ri - e * e + dot * dot)
    return inside, ro - inside


def cpu_interfaces(a, level):
    # Independent vectorized bracket discovery, then interpolate both radii directly in mm.
    r0, dr, _ = STEPS
    positive = a > level
    flips = np.diff(positive.astype("i1"), axis=0)
    require(
        not np.any(positive[[0, -1]])
        and np.all((flips == 1).sum(0) == 1)
        and np.all((flips == -1).sum(0) == 1),
        "expected one completely enclosed segment per synthetic ray",
    )

    def at(k):
        return np.take_along_axis(a, k[None], axis=0)[0].astype("f8")

    def edge(sign):
        k = (flips == sign).argmax(0)
        lo = at(k)
        hi = at(k + 1)
        return r0 + dr * k + dr * (level - lo) / (hi - lo)

    return edge(1), edge(-1)


def qualify(base, dest):
    base.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    w, h, d = DIMS
    grid = spec(base / "tube-grid.txt", DIMS, (0, 0, 0), (0, 0, 1), (1, 0, 0), STEPS)
    results = {}
    series = {}
    probes = {}
    for name, part in CASES.items():
        scan = base / name
        g = project(scan, *part)
        atlas = base / (name + "-atlas.f32")
        wall = base / (name + ".wall")
        rec = reconstruct(scan, atlas, 512, 26, 32, 16, cylinder=grid)
        require_accounting(rec, scan, atlas)
        m = measure(atlas, grid, wall, MU / 2, sector_cols=15)
        save_json(base / (name + ".json"), m)
        least = m["min_valid_point"]
        probes[name] = probe(atlas, wall, m, least["row"], least["column"])
        digest = sha(wall)
        small = measure(atlas, grid, base / "low.wall", MU / 2, budget=1, sector_cols=15)
        require(
            small["tiles"] > 1
            and m["tiles"] == 1
            and sha(base / "low.wall") == digest
            and small["thinnest_complete_sector"] == m["thinnest_complete_sector"],
            "1 MiB replay differs",
        )
        for run in [m, small]:
            require(
                run["status_counts"]["measured"] == w * h
                and run["eligible_sectors"] == w
                and run["peak_owned_bytes"] <= run["budget_bytes"]
                and run["redzones_verified"]
                and run["output_bytes"] == wall.stat().st_size
                and run["input_bytes"] == atlas.stat().st_size,
                "invalid native wall accounting",
            )
        data = np.fromfile(wall, WALL_DTYPE).reshape(h, w)
        a = np.fromfile(atlas, "<f4").reshape(d, h, w)
        left, right = cpu_interfaces(a, MU / 2)
        ti, tw = truth(part)
        ro, ri, e, deg = part
        err = data["thickness"] - tw
        sector = m["thinnest_complete_sector"]
        direction = abs((sector["azimuth_deg"] - deg + 180) % 360 - 180)
        metrics = dict(
            max_thickness_abs_mm=float(abs(err).max()),
            p95_thickness_abs_mm=float(np.percentile(abs(err), 95)),
            max_interface_abs_mm=float(
                max(abs(data["inner"] - ti).max(), abs(data["outer"] - ro).max())
            ),
            sector_direction_deg=direction,
            point_min_abs_mm=abs(m["min_valid_point"]["mm"] - (ro - ri - e)),
            cpu_abs_mm=float(
                max(
                    abs(data["inner"] - left).max(),
                    abs(data["outer"] - right).max(),
                    abs(data["thickness"] - (right - left)).max(),
                )
            ),
        )
        print(name, metrics, sector, flush=True)
        require(
            all(metrics[k] <= lim for k, lim in LIMITS.items()),
            f"{name} qualification limits failed: {metrics}",
        )
        series[name] = dict(
            mean=data["thickness"].mean(axis=0).tolist(),
            truth=tw.tolist(),
            min_error_um=(err.min(0) * 1000).tolist(),
            max_error_um=(err.max(0) * 1000).tolist(),
        )
        results[name] = dict(
            outer_mm=ro,
            bore_mm=ri,
            eccentricity_mm=e,
            truth_thinnest_deg=deg,
            truth_min_mm=ro - ri - e,
            geometry=scan.with_suffix(".geom").read_text().strip(),
            views=g.a,
            projections_sha256=sha(scan.with_suffix(".f32")),
            atlas_sha256=sha(atlas),
            wall_sha256=digest,
            native_wall=m,
            one_mib_tiles=small["tiles"],
            one_mib_byte_identical=True,
            native_reconstruction=rec,
            errors=metrics,
        )
        if name == "near-seam":
            # Change the actual reconstruction frame, not merely the labels on the output.
            quarter = spec(base / "turned-grid.txt", DIMS, (0, 0, 0), (0, 0, 1), (0, 1, 0), STEPS)
            turned = base / "turned.f32"
            turnout = base / "turned.wall"
            tm = reconstruct(scan, turned, 512, 26, 32, 16, cylinder=quarter)
            require_accounting(tm, scan, turned)
            wm = measure(turned, quarter, turnout, MU / 2, sector_cols=15)
            require(
                turnout.read_bytes() == np.roll(data, -w // 4, axis=1).tobytes()
                and wm["thinnest_complete_sector"]["column"] == (sector["column"] - w // 4) % w,
                "quarter-turn wall result changed",
            )
            results[name]["quarter_turn"] = dict(
                bitwise_equal=True,
                new_seam_deg=90,
                new_weak_sector_deg=wm["thinnest_complete_sector"]["azimuth_deg"],
            )
    scan = base / "lined-tube"
    project(scan, *CASES["near-seam"], liner=True)
    atlas = base / "lined-atlas.f32"
    out = base / "lined-tube.wall"
    rec = reconstruct(scan, atlas, 512, 26, 32, 16, cylinder=grid)
    require_accounting(rec, scan, atlas)
    lm = measure(atlas, grid, out, MU / 2, sector_cols=15)
    ld = np.fromfile(out, WALL_DTYPE)
    require(
        lm["status_counts"]["multiple"] == w * h
        and lm["min_valid_point"] is None
        and lm["thinnest_complete_sector"] is None
        and np.all(ld["status"] == STATUS.index("multiple"))
        and all(np.isnan(ld[f]).all() for f in ("inner", "outer", "thickness")),
        "lined tube must be rejected, not measured",
    )
    negative = dict(
        fixture="near-seam tube with an air gap and a second concentric wall at radii 3.9 to 4.4 mm",
        projections_sha256=sha(scan.with_suffix(".f32")),
        wall_sha256=sha(out),
        native_wall=lm,
    )
    print("Lined tube rejected:", lm["status_counts"], flush=True)
    probes["lined-tube"] = probe(atlas, out, lm, h // 2, 358)
    save_json(dest / "radial-wall-probes.json", probes)
    save_json(base / "lined-tube.json", lm)

    draw_figure(dest / "radial-wall.png", results, series, negative)
    save_json(
        dest / "radial-wall.json",
        dict(
            protocol="Two simulated finite tubes, ideal calibrated attenuation; one global half-contrast threshold; not physical accuracy.",
            manifest=grid.read_text().splitlines(),
            binary_sha256={n: sha(BUILD / n) for n in ("cone-fdk.exe", "kernels.ptx")},
            limits=LIMITS,
            cases=results,
            negative_control=negative,
        ),
    )
    return results, series


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports", type=Path, default=REPORT)
    p.add_argument("--keep", type=Path, help="keep raw simulated scans, atlas and wall maps here")
    args = p.parse_args()
    with TemporaryDirectory() as t:
        qualify(args.keep or Path(t), args.reports)
