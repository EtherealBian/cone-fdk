"""Qualify two new tube wall geometries from separate cone projections, not altered wall maps."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from compare_native import sha
from edge_review import run as edge_run
from loss_oracle import cpu_labels, unpack
from loss_phantom import DIMS, MU, STEPS
from region_batch import run as batch_run
from region_console import export
from region_review import run as region_run
from remount_loss import run as inspect
from remount_loss import write_doc
from scipy.optimize import linear_sum_assignment
from specimen_phantom import TUBES
from validate_remount import FIXTURE, GRID
from validate_roi import require
from validate_witness import BODY
from wall import WALL_DTYPE

# Fixed before detector-to-wall runs. No fitted/intensity-scaled/shifted masks or ignored edge cells.
GATE, RADIUS = 0.3, 9.4
LEVELS = [0.018, 0.020, 0.022]
LIMITS = dict(
    min_component_iou=0.85,
    max_relative_area_error=0.15,
    nominal_peak_abs_mm=0.065,
    all_pairs_peak_abs_mm=0.12,
)


def design_fields(tube, changed):
    if not changed:
        return {}
    w, h, _ = DIMS
    a = np.arange(w) * 2 * np.pi / w
    z = ((np.arange(h) - (h - 1) / 2) * STEPS[2])[:, None]
    out = {}
    phi, depth, lo, hi = tube.flat
    # Outward radial ray stops at the plane or the nominal OD, whichever it reaches first.
    dot = np.cos(a - np.deg2rad(phi))
    od = np.full_like(dot, tube.outer)
    sel = dot > 0
    od[sel] = np.minimum(tube.outer, (tube.outer - depth) / dot[sel])
    out["OD flat"] = (tube.outer - od) * ((z >= lo) & (z <= hi))
    if tube.pocket:
        phi, e, r, lo, hi = tube.pocket
        # Intersect an outward material ray with the cutter circle. No detector-ray or wall code.
        sep = a - np.deg2rad(phi)
        intercept = e * np.cos(sep)
        disc = r * r - (e * np.sin(sep)) ** 2
        reach = np.where(disc >= 0, intercept + np.sqrt(np.maximum(0.0, disc)), 0.0)
        out["ID pocket"] = np.maximum(0.0, reach - tube.inner) * ((z >= lo) & (z <= hi))
    return out


def metrics(fields, lab, scenario, nominal):
    regs = scenario["regions"]
    require(not np.any(lab < 0) and not scenario["unknown_cells"], "unknown coverage")
    require(len(regs) == len(fields), "wrong number of measured regions")
    if not fields:
        return []
    masks = [v >= GATE for v in fields.values()]
    meas = [lab == r["id"] for r in regs]
    iou = np.array([[np.sum(t & m) / np.sum(t | m) for m in meas] for t in masks])
    ti, mi = linear_sum_assignment(-iou)
    matched = []
    for i, j in zip(ti, mi, strict=True):
        name = list(fields)[i]
        t, m = masks[i], meas[j]
        r = regs[j]
        field = fields[name]
        design_seam = bool(np.any(t[:, 0] & t[:, -1]))
        area = float((m.sum() - t.sum()) / t.sum())
        diff = float(r["peak_loss_mm"] - field.max())
        require(
            iou[i, j] >= LIMITS["min_component_iou"]
            and abs(area) <= LIMITS["max_relative_area_error"]
            and abs(diff) <= LIMITS["nominal_peak_abs_mm" if nominal else "all_pairs_peak_abs_mm"]
            and r["crosses_seam"] == design_seam
            and not r["adjacent_unknown"],
            f"{name}: overlap, area, peak or seam outside predeclared gate: IoU={iou[i, j]}, area={area}, peak error={diff}",
        )
        matched.append(
            dict(
                feature=name,
                region=r["id"],
                design_cells=int(t.sum()),
                measured_cells=int(m.sum()),
                iou=float(iou[i, j]),
                area_signed_relative_error=area,
                design_peak_mm=float(field.max()),
                measured_peak_mm=r["peak_loss_mm"],
                signed_peak_error_mm=diff,
                design_crosses_seam=design_seam,
            )
        )
    return matched


def qualify(scans, work):
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    require(
        json.loads((scans / "design.json").read_bytes())
        == json.loads(json.dumps({k: asdict(v) for k, v in TUBES.items()})),
        "wrong scan design",
    )
    cases = {}
    entries = []
    source_hashes = set()
    w, h, _ = DIMS
    for tube_name, kind in [
        ("seam-flat", "change"),
        ("dual-band", "change"),
        ("seam-flat", "repeat"),
    ]:
        tag = tube_name + "-" + kind
        root = work / tag
        root.mkdir()
        tube = TUBES[tube_name]
        prefix = scans / tube_name
        parent = root / "witness"
        edge = root / "edge"
        region = root / "regions"
        # Measurement receives only projections, normal calibration and fixed process settings.
        log = inspect(
            prefix / "noisy-reference",
            prefix / ("noisy-" + kind),
            FIXTURE,
            GRID,
            parent,
            level=MU / 2,
            threshold=GATE,
            radius=RADIUS,
            witness=BODY,
        )
        require(
            log["status"] == "compared"
            and all(
                x["status"] == "pass"
                for scan in log["scans"].values()
                for x in scan["body_witnesses"]
            ),
            "witness or coverage gate",
        )
        pin = sha(parent / "manifest.json")
        e = edge_run(parent, edge, LEVELS, parent_sha256=pin)
        ep = sha(edge / "manifest.json")
        r = region_run(edge, region, RADIUS, ep)
        require(
            r["incomplete_pair_count"] == 0 and len(r["scenarios"]) == 9,
            "missing global settings",
        )
        fields = design_fields(tube, kind == "change")
        rows = []
        for s in r["scenarios"]:
            i, j = s["pair"]
            a, b = (
                np.fromfile(edge / f"{role}-{lev:02}.wall", WALL_DTYPE).reshape(h, w)
                for role, lev in [("reference", i), ("inspected", j)]
            )
            lab = cpu_labels(a, b, GATE)
            digest = hashlib.sha256(lab.astype("<i4").tobytes()).hexdigest()
            require(digest == s["labels_i32le_sha256"], "independent CPU flood fill differs")
            nominal = [i, j] == r["nominal_pair"]
            if nominal:
                np.testing.assert_array_equal(
                    lab, unpack(json.loads((parent / "material-loss.json").read_bytes()))
                )
            rows.append(
                dict(
                    pair=[i, j],
                    components=metrics(fields, lab, s, nominal),
                    labels_sha256=digest,
                    observed_flagged_cells=s["observed_flagged_cells"],
                )
            )
        hashes = {
            role: log["inputs_sha256"][role + ".f32"] for role in ["reference", "inspected"]
        }
        source_hashes.update(hashes.values())
        cases[tag] = dict(
            shape_ha=[h, w],
            tube=tube_name,
            method="paired global thresholds",
            projection_sha256=hashes,
            witness_manifest_sha256=pin,
            edge_manifest_sha256=ep,
            region_manifest_sha256=sha(region / "manifest.json"),
            independent_pairs=e["independent_pairs"],
            rows=rows,
        )
        entries.append(
            dict(
                id=tag, kind="edge", job=tag + "/edge", manifest_sha256=ep, area_radius_mm=RADIUS
            )
        )
        print(tag, "PASS", flush=True)
    require(
        len(source_hashes) == 5,
        "expected five distinct noisy detector acquisitions; only the seam reference repeats",
    )
    recipe = work / "recipe.json"
    write_doc(recipe, dict(format="CFDK-region-batch-plan-v1", entries=entries))
    batch = work / "batch"
    m, fresh = batch_run(recipe, batch)
    m2, reused = batch_run(recipe, batch, resume=True)
    require(
        m == m2
        and fresh["computed"] == 3
        and reused["reused"] == 3
        and m["counts"]["complete_coverage"] == 3,
        "batch resume",
    )
    require(
        all(
            row["result_sha256"] == cases[row["id"]]["region_manifest_sha256"]
            for row in m["entries"]
        ),
        "batch vs individual",
    )
    ui = export(recipe, batch, work / "console", sha(batch / "manifest.json"))
    require(ui["maps"] == 27, "missing offline cases")
    doc = dict(
        protocol="Two distinct simulated tube geometries, five noisy detector acquisitions, three witnessed comparisons. One reference is shared only by seam defect and repeat. All nine global levels paired without registration to truth. Matching by maximum total IoU, complete grid, fixed 9.4 mm area datum. Not industrial specimen accuracy.",
        limits=LIMITS,
        levels=LEVELS,
        loss_gate_mm=GATE,
        area_radius_mm=RADIUS,
        designs={k: asdict(v) for k, v in TUBES.items()},
        sources_sha256={
            n: sha(Path(__file__).with_name(n))
            for n in ("specimen_phantom.py", "validate_specimens.py")
        },
        cases=cases,
        console=ui,
    )
    write_doc(work / "qualification.json", doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scans", type=Path)
    p.add_argument("work", type=Path)
    qualify(**vars(p.parse_args()))
