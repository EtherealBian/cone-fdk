"""Detector-to-defect validation: truth poses are used only here and in the ray generator."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, REPORT, ROOT, reconstruct
from compare_native import sha
from loss import inspect_loss
from loss_oracle import cpu_labels, unpack
from loss_phantom import DIMS, FLAT, MU, PHOTONS, RELIEF, RO, design
from loss_probe import from_bundle, write_demo
from remount_datum import candidates, read_calibration
from remount_figure import draw
from remount_loss import run, write_doc
from remount_phantom import DEEP_FLAT, POSES, generate, pose
from validate_roi import require
from validate_stack import require_accounting
from wall import WALL_DTYPE, measure

FIXTURE = ROOT / "fixtures/remount-fixture.json"
GRID = ROOT / "fixtures/remount-material.grid"
LIMITS = dict(
    max_rotation_deg=0.07,
    max_translation_mm=0.015,
    max_off_marker_error_mm=0.025,
    min_iou=0.90,
    max_peak_error_mm=0.05,
    max_unchanged_delta_mm=0.06,
)
GATE = 0.30


def readwalls(job):
    w, h, _ = DIMS
    return [
        np.fromfile(job / role / "material.wall", WALL_DTYPE).reshape(h, w)
        for role in ["reference", "inspected"]
    ]


def check_pose(fit, mount):
    r, t = pose(mount)
    f = np.asarray(fit["rotation"])
    s = np.array(fit["translation_mm"])
    phi = np.arange(24) * 2 * np.pi / 24
    # A separate target grid covers the OD at both inspection ends and at mid-height; not used to fit anything.
    targets = np.vstack(
        [np.c_[RO * np.cos(phi), RO * np.sin(phi), phi * 0 + z] for z in [-4, 0, 4]]
    )
    err = dict(
        rotation_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(f.T @ r) - 1) / 2, -1, 1)))),
        translation_mm=float(np.linalg.norm(s - t)),
        off_marker_72_point_max_mm=float(
            np.linalg.norm(targets @ (f - r).T + s - t, axis=1).max()
        ),
        max_bead_fit_residual_mm=fit["max_residual_mm"],
    )
    require(
        err["rotation_deg"] <= LIMITS["max_rotation_deg"]
        and err["translation_mm"] <= LIMITS["max_translation_mm"]
        and err["off_marker_72_point_max_mm"] <= LIMITS["max_off_marker_error_mm"],
        "pose gate: " + str(err),
    )
    return err


def check_scout(scan, job):
    cfg = json.loads(FIXTURE.read_bytes())
    _, radius, level, _, n, fov = read_calibration(cfg)
    full = job.parent / "full-scout.f32"
    m = reconstruct(scan, full, n, fov, 96, 16)
    require_accounting(m, scan, full)
    a = np.fromfile(full, "<f4").reshape(n, n, n)
    info = json.loads((job / "manifest.json").read_bytes())["scans"]["reference"]
    _, _, lo, _, _, depth = info["scout_roi_xyz_nxyz"]
    raw = (job / "reference/scout.f32").read_bytes()
    require(
        lo > 0
        and a[lo : lo + depth].tobytes() == raw
        and info["candidates"] == candidates(a, fov, level, radius),
        "scout crop changed payload or bead centers",
    )
    return dict(
        full_bytes=a.nbytes,
        roi_bytes=len(raw),
        full_slices=n,
        roi_slices=depth,
        retained_voxels_bitwise_equal=True,
        bead_centers_identical=True,
    )


def scanner_only(scans, folder):
    folder.mkdir()
    for key, name in [("reference", "reference"), ("inspected", "repeat")]:
        dst = folder / key
        dst.mkdir()
        atlas = dst / "shells"
        m = reconstruct(scans / name, atlas, 512, 32, 96, 16, cylinder=GRID)
        require_accounting(m, scans / name, atlas)
        measure(atlas, GRID, dst / "material.wall", MU / 2)
    out = folder / "scanner-only.json"
    inspect_loss(
        folder / "reference/material.wall",
        GRID,
        folder / "inspected/material.wall",
        GRID,
        out,
        GATE,
        RO,
    )
    doc = json.loads(out.read_bytes())
    a, b = readwalls(folder)
    np.testing.assert_array_equal(unpack(doc), cpu_labels(a, b, GATE))
    require(doc["coverage"]["flagged"] > 0, "fixed scanner control must reveal false change")
    return dict(
        regions=len(doc["regions"]),
        coverage=doc["coverage"],
        peak_mm=max(r["peak"]["loss_mm"] for r in doc["regions"]),
        report_sha256=sha(out),
    ), a["thickness"] - b["thickness"]


def qualify(base, dest, *, existing_scans=None):
    base.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    scans = existing_scans or base / "scans"
    if existing_scans is None:
        generate(scans)
    results = {}
    maps = {}
    scout = None
    for part in ["repeat", "changed", "deepened", "noisy-repeat", "noisy-changed", "missing"]:
        ref = "noisy-reference" if part.startswith("noisy") else "reference"
        job = base / part
        log = run(
            scans / ref, scans / part, FIXTURE, GRID, job, level=MU / 2, threshold=GATE, radius=RO
        )
        entry = results[part] = dict(
            status=log["status"],
            manifest_sha256=sha(job / "manifest.json"),
            reference_projection_sha256=log["inputs_sha256"]["reference.f32"],
            inspected_projection_sha256=log["inputs_sha256"]["inspected.f32"],
        )
        if part == "missing":
            require(
                log["status"] == "datum_rejected"
                and log["rejected_scan"] == "inspected"
                and log["reason"] == "candidate_count:3"
                and log["loss"] is None
                and not (job / "material-loss.json").exists()
                and not list(job.rglob("*.wall")),
                "invalid datum published a comparison",
            )
            entry["reason"] = log["reason"]
            continue
        require(
            log["status"] == "compared" and log["loss"]["xyz_frame"] == "material",
            "pose or coverage rejected: " + part,
        )
        entry["pose_errors"] = {
            k: check_pose(log["scans"][k]["material_to_scanner"], mount)
            for k, mount in [("reference", "first"), ("inspected", "second")]
        }
        if scout is None:
            scout = check_scout(scans / ref, job)
        a, b = readwalls(job)
        delta = a["thickness"] - b["thickness"]
        maps[part] = delta
        doc = json.loads((job / "material-loss.json").read_bytes())
        labels = unpack(doc)
        np.testing.assert_array_equal(labels, cpu_labels(a, b, GATE))
        regs = doc["regions"]
        coverage = doc["coverage"]
        require(
            coverage["paired_valid"] == delta.size and not coverage["unknown"],
            "unknown in fixture map",
        )
        entry.update(
            coverage=coverage,
            region_count=len(regs),
            max_abs_thickness_change_mm=float(abs(delta).max()),
            loss_sha256=sha(job / "material-loss.json"),
        )
        if "repeat" in part:
            require(
                not regs and abs(delta).max() <= LIMITS["max_unchanged_delta_mm"],
                "false change in marked repeat",
            )
            continue
        deep = part == "deepened"
        ideal = (
            design(DEEP_FLAT, None)[0] - design(FLAT, None)[0]
            if deep
            else design(None, RELIEF)[1]
        )
        truth = ideal >= GATE
        pred = labels > 0
        require(
            len(regs) == 1
            and regs[0]["crosses_seam"] == deep
            and not (regs[0]["touches_axial_edge"] or regs[0]["touches_unknown"]),
            "incorrect component topology",
        )
        pk = regs[0]["peak"]
        peak_diff = abs(pk["loss_mm"] - ideal.max())
        iou = (pred & truth).sum() / (pred | truth).sum()
        y, x = pk["row"], pk["column"]
        require(
            truth[y, x]
            and iou >= LIMITS["min_iou"]
            and peak_diff <= LIMITS["max_peak_error_mm"]
            and abs(pk["inner_growth_mm" if deep else "outer_recession_mm"]) < 0.025,
            "footprint, peak or wrong-surface gate failed: " + part,
        )
        if not deep:
            require(abs(pk["azimuth_deg"] - RELIEF[0]) <= 3, "wrong azimuth")
        row = DIMS[1] // 2 - 9  # z=-1.8 mm, on the flat, directly on its normal
        if deep:
            require(
                abs(delta[row, 0] - (DEEP_FLAT[1] - FLAT[1])) < 0.025,
                "old loss was counted twice",
            )
            entry["flat_normal_delta_mm"] = float(delta[row, 0])
        entry.update(
            footprint_iou=float(iou),
            peak_mm=pk["loss_mm"],
            design_radial_peak_mm=float(ideal.max()),
            peak_error_mm=float(peak_diff),
            peak_azimuth_deg=pk["azimuth_deg"],
            surface="outer" if deep else "inner",
        )
        mapped = log["loss"]["peak_correspondences"][0]
        p = np.array(mapped["material_xyz_mm"])
        require(mapped["id"] == regs[0]["id"], "unbound peak")
        for key, mount in [("reference", "first"), ("inspected", "second")]:
            r, t = pose(mount)
            require(
                np.linalg.norm(r @ p + t - mapped[key + "_scanner_xyz_mm"])
                < LIMITS["max_off_marker_error_mm"],
                "peak scanner XYZ",
            )
        if deep:
            pack = dest / "remount-deepened.zip"
            write_demo(
                pack,
                GRID,
                job / "reference/material.wall",
                job / "inspected/material.wall",
                job / "material-loss.json",
            )
            require(from_bundle(pack, y, x)["is_reported_peak"], "offline peak replay")
            (dest / "remount-example-job.json").write_bytes((job / "manifest.json").read_bytes())
    ctrl, maps["scanner-only"] = scanner_only(scans, base / "unregistered")

    draw(dest / "remount-loss.png", maps, ctrl, results)
    report = dict(
        protocol="Simulated fixture-marked finite annulus, existing outer flat, independently ray-traced poses. No physical specimen.",
        binary_sha256={k: sha(BUILD / k) for k in ["cone-fdk.exe", "kernels.ptx"]},
        limits=LIMITS,
        loss_threshold_mm=GATE,
        fixture_sha256=sha(FIXTURE),
        material_grid_sha256=sha(GRID),
        generating_extrinsic_zyx_deg_and_translation_mm=POSES,
        noise=dict(
            photons_per_air_bin=PHOTONS, seeds=dict(reference=3671, repeat=15731, changed=85411)
        ),
        scout=scout,
        unchanged_fixed_scanner_negative_control=ctrl,
        cases=results,
        example_sha256={
            p: sha(dest / p)
            for p in ["remount-deepened.zip", "remount-example-job.json", "remount-loss.png"]
        },
    )
    write_doc(dest / "remount-loss.json", report)
    print("PASS: fixture remount inspection", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--keep", type=Path)
    cli.add_argument("--reports", type=Path, default=REPORT)
    args = cli.parse_args()
    with TemporaryDirectory() as tmp:
        qualify(args.keep or Path(tmp), args.reports)
