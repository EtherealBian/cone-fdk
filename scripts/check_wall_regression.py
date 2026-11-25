"""Compare full acceptance and a second independent wall run with the published cylinder baseline."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from common import REPORT
from compare_native import sha
from validate_roi import require
from validate_stack import save_json


def verify(q, replay, baseline):
    def load(name):
        return json.loads((q / (name + ".json")).read_bytes())

    old = json.loads(baseline.read_bytes())
    tests = ET.parse(q / "native-tests.xml").find("testsuite").attrib
    junit = {k: int(tests[k]) for k in ("tests", "failures", "errors", "skipped")}
    require(
        (q / "RESULTS.md").is_file()
        and junit["tests"] >= 140
        and not any(junit[k] for k in ("failures", "errors", "skipped")),
        "full report or clean test suite missing",
    )
    full = {f"analytic{k}": v["sha256"]["native"] for k, v in load("phantom").items()}
    full.update(
        walnut512=load("walnut-full")["output_sha256"], stress1296=load("oversize")["sha256"]
    )
    planes = {k: v["sha256"] for k, v in load("direct-planes")["planes"].items()}
    stack = load("oblique-stack")
    stacks = dict(
        walnut_payload_sha256=stack["stack_sha256"],
        simulated_object_payload_sha256=load("oblique-stack-phantom")["stack_sha256"],
        nrrd_file_sha256=stack["nrrd"]["file_sha256"],
    )
    pack = load("roi-planner-walnut")["reference_pack_sha256"]
    roi = load("roi-walnut")["fine_grid_2048"]["roi_sha256"]
    require(
        full == old["previous_full_volume_hashes_reproduced"]
        and planes == old["previous_plane_hashes_reproduced"]
        and stacks == old["previous_stack_hashes_reproduced"]
        and roi == old["previous_roi2048_sha256"]
        and pack == old["previous_six_roi_sha256"]
        and all(r["packed"]["sha256"] == pack for r in load("multi-roi-walnut")["pairs"]),
        "legacy output changed",
    )
    cyl = old["new_cylinder"]
    measured = load("cylinder-walnut")["payload_sha256"]
    simulated = load("cylinder-phantom")["sha256"]
    require(
        measured == cyl["walnut_sha256"]
        and simulated == cyl["phantom_sha256"]
        and load("roi-planner-walnut")["time_summary"]["speed_gate_passed"],
        "cylinder output changed or ROI planner gate failed",
    )
    wall = load("radial-wall")
    second = json.loads((replay / "radial-wall.json").read_bytes())
    build = load("build-repro")
    require(
        build["identical"]
        and wall["binary_sha256"]
        == second["binary_sha256"]
        == {"cone-fdk.exe": build["executable_sha256"], "kernels.ptx": build["ptx_sha256"]},
        "build mismatch",
    )
    require(
        wall["cases"].keys() == second["cases"].keys()
        and wall["negative_control"] == second["negative_control"],
        "case set or negative control changed",
    )
    for name, r in wall["cases"].items():
        keys = (
            "projections_sha256",
            "atlas_sha256",
            "wall_sha256",
            "native_wall",
            "errors",
            "quarter_turn",
        )
        require(
            all(r.get(k) == second["cases"][name].get(k) for k in keys),
            "wall replay changed: " + name,
        )
    extras = {n: sha(q / n) for n in ("radial-wall.png", "radial-wall-probes.json")}
    require(all(v == sha(replay / n) for n, v in extras.items()), "evidence replay changed")
    return dict(
        protocol="Two identical builds; full Run-Acceptance.ps1; independent second wall qualification. Hashes below compared to cylinder-regression.json.",
        gpu=wall["cases"]["near-seam"]["native_wall"]["gpu"],
        pytest=junit,
        build=build,
        legacy_full=full,
        legacy_planes=planes,
        legacy_stacks=stacks,
        legacy_roi_2048=roi,
        legacy_six_roi_pack=pack,
        legacy_cylinder=dict(walnut=measured, analytic=simulated),
        roi_planner_speed_gate=True,
        wall_replays_equal=True,
        evidence_sha256=extras,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("full", type=Path)
    p.add_argument("second", type=Path)
    p.add_argument("--baseline", type=Path, default=REPORT / "cylinder-regression.json")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    save_json(args.out, verify(args.full, args.second, args.baseline))
    print("PASS: legacy hashes and independent wall replay match.")
