"""Qualify global edge scenarios against the native CUDA loss inspector and a separate flood fill."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from common import BUILD
from compare_native import sha
from loss import inspect_loss
from loss_oracle import cpu_labels, unpack
from region_review import run
from remount_loss import write_doc
from validate_roi import require
from wall import WALL_DTYPE


def qualify(reviews, work, dest, trusted):
    work.mkdir(parents=True, exist_ok=False)
    dest.mkdir(parents=True, exist_ok=True)
    records = {}
    trusted_cases = json.loads(trusted.read_bytes())["cases"]
    tools = {n: sha(BUILD / n) for n in ("cone-fdk.exe", "kernels.ptx")}
    for name in ("repeat", "noisy-change", "invalid-edge-control"):
        parent = reviews / name
        grid = parent / "material.grid"
        expected = trusted_cases[name]["review_manifest_sha256"]
        src = json.loads((parent / "manifest.json").read_bytes())
        require(
            all(src["inputs_sha256"][n] == d for n, d in tools.items()), "different native build"
        )
        doc = run(parent, work / name, 9.4, expected)
        nominal = doc["nominal_pair"][0]
        order = sorted(doc["scenarios"], key=lambda s: s["pair"] != [nominal, nominal])
        shape = tuple(doc["shape_ha"])
        ng = None
        probes = []
        for s in order:
            i, j = s["pair"]
            a, b = (parent / n for n in (f"reference-{i:02}.wall", f"inspected-{j:02}.wall"))
            dest_path = work / name / f"native-{i}-{j}.json"
            stats = inspect_loss(
                a, grid, b, grid, dest_path, doc["loss_gate_mm"], doc["area_radius_mm"], budget=1
            )
            native = json.loads(dest_path.read_bytes())
            require(
                stats["redzones_verified"]
                and 0 < stats["peak_owned_bytes"] <= stats["budget_bytes"] == 1 << 20
                and stats["output_bytes"] == dest_path.stat().st_size
                and stats["input_bytes_read"] == 4 * a.stat().st_size,
                "native accounting",
            )
            lab = unpack(native)
            oracle = cpu_labels(
                *(np.fromfile(p, WALL_DTYPE).reshape(shape) for p in (a, b)), doc["loss_gate_mm"]
            )
            np.testing.assert_array_equal(lab, oracle)
            lab = np.maximum(lab, -1).astype("<i4")
            require(
                hashlib.sha256(lab.tobytes()).hexdigest() == s["labels_i32le_sha256"],
                "canonical labels differ",
            )
            if ng is None:
                ng = lab
            # Count intersecting ID pairs independently (only on actual flagged cells).
            overlaps = {}
            for r, c in np.argwhere((lab > 0) & (ng > 0)):
                key = int(ng[r, c]), int(lab[r, c])
                overlaps[key] = overlaps.get(key, 0) + 1
            require(
                s["nominal_overlap"] == [[*k, overlaps[k]] for k in sorted(overlaps)],
                "overlap census",
            )
            require(
                s["region_count"] == len(native["regions"])
                and s["unknown_cells"] == native["coverage"]["unknown"]
                and s["observed_flagged_cells"] == native["coverage"]["flagged"],
                "coverage or count",
            )
            for x, y in zip(s["regions"], native["regions"], strict=True):
                p = y["peak"]
                require(
                    all(x[k] == y[k] for k in ("id", "cells", "crosses_seam"))
                    and x["peak_cell"] == [p["row"], p["column"]]
                    and x["peak_loss_mm"] == p["loss_mm"]
                    and abs(x["area_mm2"] - y["area_mm2"]) < 1e-12
                    and x["adjacent_unknown"] == y["touches_unknown"]
                    and x["touches_axial_end"] == y["touches_axial_edge"],
                    "component properties",
                )
            probes.append(
                dict(
                    pair=[i, j],
                    native_report_sha256=sha(dest_path),
                    labels_i32le_sha256=s["labels_i32le_sha256"],
                )
            )
        e = doc["observed_extrema"]["observed_flagged_cells"]
        if name == "noisy-change":
            require(
                all(s["region_count"] == 1 for s in order)
                and [e["min"]["value"], e["max"]["value"]] == [253, 275]
                and doc["incomplete_pair_count"] == 0,
                "relief scenario bounds",
            )
            write_doc(dest / "region-noisy-change.json", doc)
        elif name == "repeat":
            require(
                e["max"]["value"] == 0 and not doc["incomplete_pair_count"],
                "repeat false positive or coverage",
            )
        else:
            require(
                doc["incomplete_pair_count"] == 3
                and order[0]["unknown_cells"] == 0
                and all(s["unknown_cells"] > 0 for s in order[1:]),
                "invalid edge control lost coverage",
            )
        records[name] = dict(
            edge_sha256=expected,
            region_sha256=sha(work / name / "manifest.json"),
            pair_cells=len(order) * int(np.prod(shape)),
            incomplete_pairs=doc["incomplete_pair_count"],
            cases=probes,
        )
    write_doc(
        dest / "region-qualification.json",
        dict(
            protocol="Same saved wall evidence, explicit global pairs. Periodic component IDs checked cell by cell against native CUDA loss and a separate CPU flood fill; this does not validate physical accuracy.",
            native_sha256=tools,
            cases=records,
        ),
    )
    print(
        "PASS: all global pairs, component labels, overlap links, coverage and peak measurements",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reviews", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("--reports", type=Path, dest="dest", required=True)
    p.add_argument(
        "--trusted",
        type=Path,
        required=True,
        help="independently obtained edge-qualification.json",
    )
    qualify(**vars(p.parse_args()))
