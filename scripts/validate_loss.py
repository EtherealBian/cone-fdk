"""End-to-end local wall loss qualification, from fresh analytic X-rays to native region records."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, REPORT, reconstruct
from compare_native import sha
from loss import inspect_loss
from loss_figure import draw
from loss_oracle import cpu_labels, unpack
from loss_phantom import (
    CASES,
    DIMS,
    MU,
    NOISE_SEEDS,
    PHOTONS,
    RO,
    STEPS,
    counting_scan,
    design,
    project,
)
from loss_probe import from_bundle, write_demo
from loss_scale import qualify_scale
from validate_cylinder import spec
from validate_roi import require
from validate_stack import require_accounting, save_json
from wall import WALL_DTYPE, measure

THRESHOLD = 0.30
LIMITS = {"min_iou": 0.90, "max_area_rel": 0.12, "peak_abs_mm": 0.07, "max_angle_error_deg": 4.0}


NOISY_LIMITS = {
    "min_iou": 0.88,
    "max_area_rel": 0.14,
    "peak_abs_mm": 0.085,
    "max_angle_error_deg": 6.0,
}


def qualify(base, dest):
    base.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    w, h, _ = DIMS
    for name, tools in CASES.items():
        project(base / name, *tools)
    for name, (source, seed) in NOISE_SEEDS.items():
        counting_scan(base / source, base / name, seed)

    def grid(seam):
        return spec(
            base / f"seam-{seam}.grid",
            DIMS,
            (0, 0, 0),
            (0, 0, 1),
            {0: (1, 0, 0), 90: (0, 1, 0)}[seam],
            STEPS,
        )

    def make_wall(name, frame):
        scan = base / name
        atlas = base / f"{name}-{frame.stem}.atlas"
        out = atlas.with_suffix(".wall")
        rec = reconstruct(scan, atlas, 512, 26, 32, 16, cylinder=frame)
        require_accounting(rec, scan, atlas)
        m = measure(atlas, frame, out, MU / 2)
        require(m["status_counts"]["measured"] == w * h, "incomplete simulated wall")
        return out, dict(
            projections_sha256=sha(scan.with_suffix(".f32")),
            atlas_sha256=sha(atlas),
            wall_sha256=sha(out),
        )

    def read(path):
        return np.fromfile(path, WALL_DTYPE).reshape(h, w)

    results = {}
    evidence = {}
    maps = {}
    deviations = {}
    scenarios = [
        (0, "reference", ["outer-flat", "dual-surface", "subthreshold"]),
        (90, "reference", ["dual-surface"]),
        (0, "noisy-reference", ["noisy-clean", "noisy-dual"]),
    ]
    for seam, ref_name, names in scenarios:
        frame = grid(seam)
        ref, rh = make_wall(ref_name, frame)
        a = read(ref)
        limits = NOISY_LIMITS if ref_name == "noisy-reference" else LIMITS
        for name in names:
            out, hashes = make_wall(name, frame)
            b = read(out)
            doc = base / f"{name}-{seam}.json"
            m = inspect_loss(ref, frame, out, frame, doc, THRESHOLD, RO, budget=1)
            before = doc.read_bytes()
            native = json.loads(before)
            large = inspect_loss(ref, frame, out, frame, doc, THRESHOLD, RO, budget=16)
            require(
                doc.read_bytes() == before
                and all(
                    x["peak_owned_bytes"] <= x["budget_bytes"]
                    and x["redzones_verified"]
                    and x["input_bytes_read"] == 4 * ref.stat().st_size
                    and x["output_bytes"] == len(before)
                    for x in (m, large)
                ),
                "budget replay/accounting",
            )
            labels = unpack(native)
            np.testing.assert_array_equal(labels, cpu_labels(a, b, THRESHOLD))
            require(
                not native["coverage"]["unknown"]
                and native["coverage"]["flagged"] == int((labels > 0).sum()),
                "coverage",
            )
            source_name = NOISE_SEEDS[name][0] if name in NOISE_SEEDS else name
            dent, growth = design(*CASES[source_name], seam)
            require(
                rh["projections_sha256"] != hashes["projections_sha256"],
                "control/part must not reuse reference bytes",
            )
            targets = [
                (t, side, phase)
                for t, side, phase in [(dent, "outer", 0), (growth, "inner", 134)]
                if np.any(t >= THRESHOLD)
            ]
            require(
                len(targets) == len(native["regions"]),
                f"{name}: component count mismatch: {len(native['regions'])}",
            )
            metrics = []
            seen = set()
            loss = a["thickness"] - b["thickness"]
            for ideal, side, phase in targets:
                truth = ideal >= THRESHOLD
                reg = max(
                    native["regions"], key=lambda r: np.count_nonzero((labels == r["id"]) & truth)
                )
                pred = labels == reg["id"]
                seen.add(reg["id"])
                pk = reg["peak"]
                y, x = pk["row"], pk["column"]
                iou = float((pred & truth).sum() / (pred | truth).sum())
                area_rel = abs(pred.sum() / truth.sum() - 1)
                d_err = abs(pk["loss_mm"] - float(ideal.max()))
                angle = abs((seam + pk["azimuth_deg"] - phase + 180) % 360 - 180)
                expected_area = truth.sum() * 2 * np.pi * RO / w * STEPS[2]
                s = dict(
                    surface=side,
                    id=reg["id"],
                    iou=iou,
                    area_relative_error=float(area_rel),
                    design_cell_area_mm2=float(expected_area),
                    reported_area_mm2=reg["area_mm2"],
                    design_peak_mm=float(ideal.max()),
                    reported_peak_mm=pk["loss_mm"],
                    peak_abs_error_mm=d_err,
                    peak_angle_error_deg=angle,
                    crosses_seam=reg["crosses_seam"],
                )
                print(name, seam, s, flush=True)
                require(
                    iou >= limits["min_iou"]
                    and area_rel <= limits["max_area_rel"]
                    and d_err <= limits["peak_abs_mm"]
                    and angle <= limits["max_angle_error_deg"],
                    f"{name}: design gate failed: {s}",
                )
                require(
                    reg["crosses_seam"] == (side == "outer" and seam == 0)
                    and not reg["touches_unknown"]
                    and not reg["touches_axial_edge"],
                    "region boundary flags",
                )
                field = "outer_recession_mm" if side == "outer" else "inner_growth_mm"
                require(
                    abs(pk[field] - ideal.max()) <= limits["peak_abs_mm"]
                    and abs(pk["loss_mm"] - loss[y, x]) < 1e-12
                    and abs(reg["area_mm2"] - pred.sum() * 2 * np.pi * RO / w * STEPS[2]) < 1e-12
                    and abs(reg["mean_loss_mm"] - loss[pred].mean()) < 1e-12
                    and abs(reg["min_remaining_mm"] - b["thickness"][pred].min()) < 1e-12,
                    "region measurements or surface mismatch",
                )
                metrics.append(s)
            require(len(seen) == len(native["regions"]), "non-unique matches")
            key = f"{name}-{seam}"
            maps[key] = labels > 0
            deviations[key] = loss
            results[key] = dict(
                reference=rh,
                inspected=hashes,
                report_sha256=sha(doc),
                native=m,
                second_budget=large,
                coverage=native["coverage"],
                regions=metrics,
                whole_map_max_abs_error_mm=float(abs(loss - dent - growth).max()),
                max_observed_loss_mm=float(loss.max()),
            )
            if name == "dual-surface" and seam == 0:
                evidence = native
    np.testing.assert_array_equal(
        maps["dual-surface-90"], np.roll(maps["dual-surface-0"], -w // 4, axis=1)
    )
    require(
        deviations["dual-surface-90"].tobytes()
        == np.roll(deviations["dual-surface-0"], -w // 4, axis=1).tobytes(),
        "seam change altered FP64 loss bytes",
    )
    save_json(
        dest / "wall-loss.json",
        dict(
            protocol="Analytic reference and machined tubes, ideal plus independent seeded detector counting noise; registered scanner grid; errors against the sampled design, not physical accuracy.",
            binary_sha256={n: sha(BUILD / n) for n in ("cone-fdk.exe", "kernels.ptx")},
            grid=grid(0).read_text().splitlines(),
            threshold_mm=THRESHOLD,
            limits=LIMITS,
            noisy_limits=NOISY_LIMITS,
            photons_per_air_bin=PHOTONS,
            noise_inputs_and_seeds=NOISE_SEEDS,
            quarter_turn_loss_bitwise_equal=True,
            cases=results,
        ),
    )
    # Keep the actual native bytes, not a rendered or reserialized example.
    (dest / "wall-loss-example.json").write_bytes((base / "dual-surface-0.json").read_bytes())
    bundle = dest / "wall-loss-demo.zip"
    write_demo(
        bundle,
        grid(0),
        base / "reference-seam-0.wall",
        base / "dual-surface-seam-0.wall",
        dest / "wall-loss-example.json",
    )
    for region in evidence["regions"]:
        peak = region["peak"]
        point = from_bundle(bundle, peak["row"], peak["column"])
        require(
            point["region_id"] == region["id"] and point["is_reported_peak"],
            "portable point replay failed",
        )

    qualify_scale(base, dest)
    draw(
        dest / "wall-loss.png",
        deviations["dual-surface-0"],
        evidence,
        design(*CASES["dual-surface"]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", type=Path)
    p.add_argument("--reports", type=Path, default=REPORT)
    a = p.parse_args()
    with TemporaryDirectory() as t:
        qualify(a.keep or Path(t), a.reports)
