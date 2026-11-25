"""Score the locked two-lobe study. Unmet scientific gates are retained, never tuned away."""

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
from bridge_design import SPEC_SHA256, linked, loss_field, protocol
from compare_native import sha
from edge_review_packed import run as packed
from loss import inspect_loss
from loss_oracle import cpu_labels, unpack
from loss_phantom import MU
from region_review import run as regions
from remount_loss import run as inspect
from remount_loss import write_doc
from validate_remount import FIXTURE, GRID
from validate_roi import require
from validate_witness import BODY
from wall import WALL_DTYPE


def census(labels, truth_mask, anchors):
    if (
        labels.shape != truth_mask.shape
        or truth_mask.dtype != np.bool_
        or labels.dtype.kind != "i"
    ):
        raise ValueError("incompatible labels or design mask")
    flagged = labels > 0
    a, b = (int(labels[tuple(x)]) for x in anchors)
    unknown = int(np.count_nonzero(labels < 0))
    measured = a > 0 and b > 0
    union = (flagged | truth_mask).sum()
    connected = None if not measured or unknown else a == b
    if connected is not None and linked(flagged, anchors) != connected:
        raise ValueError("labels contradict independent anchor reachability")
    return dict(
        anchor_ids=[a, b],
        anchors_both_flagged=measured,
        unknown_cells=unknown,
        connected=connected,
        footprint_iou=None
        if unknown or not union
        else float((flagged & truth_mask).sum() / union),
        observed_flagged_cells=int(flagged.sum()),
        components=len(np.unique(labels[flagged])),
    )


def assess(cases, spec):
    """Interpret only complete within-acquisition 3x3 grids; no pooling across repeats."""
    registered = [s["id"] for s in spec["scans"][1:]]
    require(set(cases) == set(registered), "case inventory")
    limits = spec["acceptance"]
    gates = {}
    for name in registered:
        c = cases[name]
        rows = c["rows"]
        pairs = [tuple(r["pair"]) for r in rows]
        require(
            sorted(pairs) == list(product(range(3), repeat=2))
            and tuple(c["nominal_pair"]) == (1, 1),
            "paired inventory",
        )
        gates[name + "/coverage-and-anchors"] = all(
            r["unknown_cells"] == limits["unknown_cells_per_pair"]
            and r["anchors_both_flagged"]
            and type(r["connected"]) is bool
            for r in rows
        )
        if name.endswith("-control"):
            want = name == "joined-control"
            gates[name + "/nine-states"] = all(r["connected"] is want for r in rows)
            n = rows[pairs.index((1, 1))]
            gates[name + "/nominal-shape-and-peak"] = (
                n["footprint_iou"] is not None
                and n["footprint_iou"] >= limits["min_control_nominal_footprint_iou"]
                and n["peak_error_mm"] is not None
                and abs(n["peak_error_mm"]) <= limits["max_control_nominal_peak_error_mm"]
            )
    gates["critical-within-acquisition-switch"] = any(
        gates[name + "/coverage-and-anchors"]
        and {r["connected"] for r in cases[name]["rows"]} == {False, True}
        for name in ("critical-a", "critical-b")
    )
    return dict(passed=all(gates.values()), gates=gates)


def qualify(scans, work, companion):
    spec = protocol()
    m = spec["measurement"]
    levels = m["edge_levels_per_mm"]
    gate = m["loss_gate_mm"]
    radius = m["area_radius_mm"]
    shape = tuple(m["shape_ha"])
    inputs = json.loads((scans / "sources.json").read_bytes())
    require(
        inputs["study_sha256"] == SPEC_SHA256
        and set(inputs["scans"]) == {s["id"] for s in spec["scans"]},
        "wrong acquisition inventory",
    )
    for key, rec in inputs["scans"].items():
        require(
            rec == {ext: sha((scans / key).with_suffix("." + ext)) for ext in ("geom", "f32")},
            "source has changed",
        )
    require(
        len({s["f32"] for s in inputs["scans"].values()}) == 5, "reused noisy projection payload"
    )
    work.mkdir(parents=True, exist_ok=False)
    cases = {}
    for scan in spec["scans"][1:]:
        name = scan["id"]
        where = work / name
        parent = where / "witness"
        edge = where / "edge"
        out = where / "regions"
        log = inspect(
            scans / "baseline",
            scans / name,
            FIXTURE,
            GRID,
            parent,
            level=MU / 2,
            threshold=gate,
            radius=radius,
            witness=BODY,
        )
        require(
            log["status"] == "compared"
            and all(
                x["status"] == "pass"
                for role in ("reference", "inspected")
                for x in log["scans"][role]["body_witnesses"]
            )
            and all(
                log["inputs_sha256"][role + ".f32"] == inputs["scans"][case]["f32"]
                for role, case in [("reference", "baseline"), ("inspected", name)]
            ),
            "registration, witness or source",
        )
        pin = sha(parent / "manifest.json")
        e = packed(parent, edge, levels, parent_sha256=pin, companion_sha256=companion)
        ep = sha(edge / "manifest.json")
        r = regions(edge, out, radius, ep)
        require(
            e["independent_pairs"] == 9
            and r["shape_ha"] == list(shape)
            and r["loss_gate_mm"] == gate
            and r["levels_per_mm"] == levels,
            "unexpected settings",
        )
        # Only after measuring from the two acquisitions do we access the design for scoring.
        truth = loss_field(spec, scan["neck_depth_mm"])
        expected = truth >= gate
        rows = []
        for s in r["scenarios"]:
            i, j = s["pair"]
            walls = [
                edge / f"{role}-{n:02}.wall" for role, n in [("reference", i), ("inspected", j)]
            ]
            native = where / f"native-{i}-{j}.json"
            grid = edge / "material.grid"
            stats = inspect_loss(walls[0], grid, walls[1], grid, native, gate, radius, budget=1)
            require(
                stats["redzones_verified"]
                and 0 < stats["peak_owned_bytes"] <= stats["budget_bytes"] == 1 << 20,
                "native accounting",
            )
            lab = unpack(json.loads(native.read_bytes()))
            oracle = cpu_labels(*(np.fromfile(w, WALL_DTYPE).reshape(shape) for w in walls), gate)
            np.testing.assert_array_equal(lab, oracle)
            digest = hashlib.sha256(np.maximum(lab, -1).astype("<i4").tobytes()).hexdigest()
            require(digest == s["labels_i32le_sha256"], "canonical labels differ")
            if s["pair"] == r["nominal_pair"]:
                np.testing.assert_array_equal(
                    lab, unpack(json.loads((parent / "material-loss.json").read_bytes()))
                )
            row = census(lab, expected, m["anchors_row_col"])
            require(
                row["components"] == s["region_count"]
                and row["observed_flagged_cells"] == s["observed_flagged_cells"]
                and row["unknown_cells"] == s["unknown_cells"],
                "coverage or component totals differ",
            )
            peak = max((v["peak_loss_mm"] for v in s["regions"]), default=None)
            rows.append(
                dict(
                    pair=[i, j],
                    **row,
                    peak_mm=peak,
                    peak_error_mm=peak - float(truth.max()) if peak is not None else None,
                    labels_i32le_sha256=digest,
                    native_sha256=sha(native),
                )
            )
        cases[name] = dict(
            witness_sha256=pin,
            edge_sha256=ep,
            region_sha256=sha(out / "manifest.json"),
            nominal_pair=r["nominal_pair"],
            rows=rows,
        )
        print(
            name,
            "connected / disconnected / indeterminate:",
            *[sum(x["connected"] is v for x in rows) for v in (True, False, None)],
            flush=True,
        )
    doc = dict(
        format="CFDK-bridge-result-v1",
        scope="One simulated geometry, one reference + four noisy inspected acquisitions. Not physical accuracy or a probability estimate.",
        protocol_sha256=SPEC_SHA256,
        acquisitions=inputs["scans"],
        sources_sha256={
            k: sha(Path(__file__).with_name(k))
            for k in ("bridge_phantom.py", "bridge_design.py", "validate_bridge.py")
        },
        companion_sha256=companion,
        cases=cases,
        acceptance=assess(cases, spec),
    )
    write_doc(work / "result.json", doc)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("scans", "work"):
        p.add_argument(k, type=Path)
    p.add_argument("--companion-sha256", required=True)
    args = p.parse_args()
    doc = qualify(args.scans, args.work, args.companion_sha256)
    print(json.dumps(doc["acceptance"]))
    raise SystemExit(0 if doc["acceptance"]["passed"] else 2)
