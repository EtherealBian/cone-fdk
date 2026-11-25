"""Inspect two fixture-marked scans in material coordinates after arbitrary rigid remounts."""

import argparse
import json
import shutil
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np
from axis_gauge import pinned_reader
from body_witness import calibration as witness_calibration
from body_witness import inspect as inspect_witness
from common import BUILD, Geometry, reconstruct
from compare_native import sha
from loss import inspect_loss
from remount_datum import (
    DatumRejected,
    candidates,
    panel_encloses,
    read_calibration,
    register,
    scout_slab,
)
from validate_stack import require_accounting
from wall import measure


def write_doc(path, doc):
    path.write_text(
        json.dumps(doc, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )


def scanner_grid(material, out, fit, geom, *, bead_centers=(), clearance=0, body=None):
    vals = material.read_text(encoding="utf-8-sig").split()
    if len(vals) != 16 or vals[0] != "CFDKCYL1":
        raise ValueError("invalid material grid")
    w, h, k = map(int, vals[1:4])
    center, axis, seam, steps = np.asarray(vals[4:], dtype="f8").reshape(4, 3)
    if (
        not np.isfinite([center, axis, seam, steps]).all()
        or not (3 <= w <= 4096 and 1 <= h <= 4096 and 2 <= k <= 4096)
        or min(steps) <= 0
        or max(abs(np.linalg.norm(axis) - 1), abs(np.linalg.norm(seam) - 1), abs(axis @ seam))
        > 1e-6
    ):
        raise ValueError("invalid material grid dimensions, spacing or frame")
    axis = axis / np.linalg.norm(axis)
    seam = seam - (axis @ seam) * axis
    seam /= np.linalg.norm(seam)
    r = np.array(fit["rotation"])
    t = np.array(fit["translation_mm"])
    center = r @ center + t
    axis = r @ axis
    seam = r @ seam

    # No fiducial may be part of the shell being inspected. Distance to the finite
    # annular sampling envelope, not merely distance to its centerline.
    exclusions = [(bead, clearance, "bead_intersects_inspection_grid") for bead in bead_centers]
    if body is not None:
        exclusions.extend(
            (r @ p + t, body.radius + body.limit, "witness_intersects_inspection_grid")
            for p in body.points
        )
    for point, separation, reason in exclusions:
        delta = np.asarray(point) - center
        axial = delta @ axis
        radial = np.linalg.norm(delta - axial * axis)
        dr = max(0, steps[0] - radial, radial - steps[0] - (k - 1) * steps[1])
        dz = max(0, abs(axial) - (h - 1) * steps[2] / 2)
        if np.hypot(dr, dz) <= separation:
            raise DatumRejected(reason)

    def support(v):
        co = v @ axis
        return abs(co) * (h - 1) * steps[2] / 2 + (steps[0] + (k - 1) * steps[1]) * np.sqrt(
            np.maximum(0, 1 - co**2)
        )

    if not panel_encloses(geom, center, support):
        raise DatumRejected("inspection_grid_outside_detector")
    out.write_text(
        "CFDKCYL1 "
        + "\n".join(" ".join(map(str, row)) for row in [(w, h, k), center, axis, seam, steps])
        + "\n",
        encoding="ascii",
        newline="\n",
    )


def run(
    reference,
    inspected,
    fixture,
    material,
    job,
    *,
    level,
    threshold,
    radius,
    budget=96,
    batch=16,
    witness=None,
):
    """A manifest commits a fresh job. A rejected registration has no loss report."""
    if not np.isfinite([level, threshold, radius]).all() or min(level, threshold, radius) <= 0:
        raise ValueError("positive finite measurement parameters required")
    sources = {
        f"{role}{ext}": scan.with_suffix(ext)
        for role, scan in [("reference", reference), ("inspected", inspected)]
        for ext in [".f32", ".geom"]
    }
    sources.update({n: BUILD / n for n in ["cone-fdk.exe", "kernels.ptx"]})
    configs = {"fixture.json": fixture, "material.grid": material}
    if witness is not None:
        configs["body-witness.json"] = witness
    sources.update(configs)
    job.mkdir(parents=True, exist_ok=False)
    with ExitStack() as held:
        for path in sources.values():
            held.enter_context(pinned_reader(path))
        for name, path in configs.items():
            shutil.copyfile(path, job / name)
        cfg = json.loads((job / "fixture.json").read_bytes())
        points, bead_radius, bead_level, limit, n, fov = read_calibration(cfg)
        body = None
        locator_sources = ["remount_loss.py", "remount_datum.py"]
        if witness is not None:
            body = witness_calibration(
                json.loads((job / "body-witness.json").read_bytes()),
                points,
                bead_radius,
                bead_level,
                fov / n,
            )
            locator_sources.append("body_witness.py")
        log = dict(
            format="CFDK-remount-job-v1" if body is None else "CFDK-remount-witness-job-v1",
            inputs_sha256={k: sha(p) for k, p in sources.items()},
            parameters=dict(
                level_per_mm=level,
                loss_mm=threshold,
                area_radius_mm=radius,
                budget_mib=budget,
                batch=batch,
            ),
            locator_software=dict(
                numpy=version("numpy"),
                scipy=version("scipy"),
                source_sha256={n: sha(Path(__file__).with_name(n)) for n in locator_sources},
            ),
            scans={},
            status="datum_rejected",
            loss=None,
        )
        if log["inputs_sha256"]["reference.f32"] == log["inputs_sha256"]["inspected.f32"]:
            raise ValueError("two distinct projection payloads required")
        try:
            for key, scan in [("reference", reference), ("inspected", inspected)]:
                geom = Geometry.read(scan.with_suffix(".geom"))
                folder = job / key
                folder.mkdir()
                info = {}
                log["scans"][key] = info
                scout = folder / "scout.f32"
                lo, depth = scout_slab(n, fov, bead_radius, geom)
                info["scout_roi_xyz_nxyz"] = [0, 0, lo, n, n, depth]
                roi = ",".join(map(str, info["scout_roi_xyz_nxyz"]))
                m = reconstruct(scan, scout, n, fov, budget, batch, roi=roi)
                require_accounting(m, scan, scout)
                info["scout_projection_passes"] = m["projection_passes"]
                volume = np.fromfile(scout, "<f4").reshape(depth, n, n)
                observed = candidates(volume, fov, bead_level, bead_radius, z0=lo)
                info["candidates"] = observed
                if not all(
                    panel_encloses(geom, np.array(b["scanner_xyz_mm"]), lambda v: bead_radius)
                    for b in observed
                ):
                    raise DatumRejected("bead_outside_detector")
                fit = register(points, [b["scanner_xyz_mm"] for b in observed], limit)
                info["material_to_scanner"] = fit
                scanner_grid(
                    job / "material.grid",
                    folder / "scanner.grid",
                    fit,
                    geom,
                    bead_centers=[b["scanner_xyz_mm"] for b in observed],
                    clearance=bead_radius + limit,
                    body=body,
                )
                if body is not None:
                    checks = inspect_witness(volume, lo, fov, geom, fit, body)
                    info["body_witnesses"] = checks
                    if any(c["status"] != "pass" for c in checks):
                        raise DatumRejected("body_witness_failed")
                del volume
            # Fit both mounts before measuring either. No resampling, optimal defect shift or relabeling.
            for key, scan in [("reference", reference), ("inspected", inspected)]:
                folder = job / key
                grid = folder / "scanner.grid"
                atlas = folder / "shells.f32"
                out = folder / "material.wall"
                m = reconstruct(scan, atlas, 512, fov, budget, batch, cylinder=grid)
                require_accounting(m, scan, atlas)
                w = measure(atlas, grid, out, level, budget=budget)
                if not (
                    w["redzones_verified"]
                    and w["peak_owned_bytes"] <= w["budget_bytes"]
                    and atlas.stat().st_size == w["input_bytes"]
                    and out.stat().st_size == w["output_bytes"]
                ):
                    raise RuntimeError("wall accounting mismatch")
                log["scans"][key].update(
                    wall_status_counts=w["status_counts"],
                    shell_projection_passes=m["projection_passes"],
                )
            grid = job / "material.grid"
            loss = job / "material-loss.json"
            m = inspect_loss(
                job / "reference/material.wall",
                grid,
                job / "inspected/material.wall",
                grid,
                loss,
                threshold,
                radius,
                budget=budget,
            )
            if not (
                m["redzones_verified"]
                and m["peak_owned_bytes"] <= m["budget_bytes"]
                and m["output_bytes"] == loss.stat().st_size
            ):
                raise RuntimeError("loss accounting mismatch")
            doc = json.loads(loss.read_bytes())
            peaks = []
            for region in doc["regions"]:
                x = np.array(region["peak"]["inspected_outer_xyz_mm"])
                peak = dict(id=region["id"], material_xyz_mm=x.tolist())
                for key, info in log["scans"].items():
                    fit = info["material_to_scanner"]
                    peak[key + "_scanner_xyz_mm"] = (
                        np.array(fit["rotation"]) @ x + fit["translation_mm"]
                    ).tolist()
                peaks.append(peak)
            log.update(
                status="compared" if not doc["coverage"]["unknown"] else "incomplete_coverage",
                loss=dict(
                    file=loss.name,
                    xyz_frame="material",
                    coverage=doc["coverage"],
                    peak_correspondences=peaks,
                ),
            )
        except DatumRejected as e:
            log.update(rejected_scan=key, reason=str(e))
        log["outputs_sha256"] = {
            p.relative_to(job).as_posix(): sha(p) for p in sorted(job.rglob("*")) if p.is_file()
        }
        temp = job / "manifest.partial.json"
        write_doc(temp, log)
        temp.replace(job / "manifest.json")
        return log


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ["reference", "inspected", "fixture", "material", "job"]:
        cli.add_argument(name, type=Path)
    for name in ["level", "threshold", "radius"]:
        cli.add_argument("--" + name, type=float, required=True)
    cli.add_argument(
        "--witness", type=Path, help="calibrated workpiece inserts; veto only, never fit"
    )
    cli.add_argument("--budget", type=int, default=96)
    cli.add_argument("--batch", type=int, default=16)
    result = run(**vars(cli.parse_args()))
    print(json.dumps({k: result.get(k) for k in ["status", "rejected_scan", "reason"]}))
    raise SystemExit(0 if result["status"] == "compared" else 2)
