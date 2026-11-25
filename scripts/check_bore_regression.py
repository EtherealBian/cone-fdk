"""Verify two gauge qualifications plus frozen pre-gauge outputs, not just an exit code."""

import argparse
import json
from pathlib import Path

from check_wall_regression import verify as verify_wall
from common import REPORT
from compare_native import sha
from validate_roi import require
from validate_stack import save_json


def verify(full, second, baseline):
    prior = verify_wall(full, second, baseline / "cylinder-regression.json")
    require(prior["pytest"]["tests"] >= 161, "missing bore tests")
    frozen = json.loads((baseline / "radial-wall.json").read_bytes())
    new = json.loads((full / "radial-wall.json").read_bytes())

    def hashes(r):
        return {
            **{k: v["wall_sha256"] for k, v in r["cases"].items()},
            "lined-tube": r["negative_control"]["wall_sha256"],
        }

    require(hashes(frozen) == hashes(new), "previous wall bytes changed")
    gauge = full / "bore-gauge.json"
    require(gauge.read_bytes() == (second / gauge.name).read_bytes(), "gauge replay changed")
    b = json.loads(gauge.read_bytes())
    build = prior["build"]
    require(
        b["binary_sha256"]
        == {"cone-fdk.exe": build["executable_sha256"], "kernels.ptx": build["ptx_sha256"]},
        "unqualified gauge build",
    )
    return dict(
        protocol="Independent clean simulation replay; full GPU acceptance; legacy volumes, ROI, planes, stack, cylinder and wall hashes compared with frozen reports.",
        build=build,
        gpu=prior["gpu"],
        pytest=prior["pytest"],
        legacy={k: v for k, v in prior.items() if k.startswith("legacy_")},
        legacy_wall=hashes(new),
        planner_gate=prior["roi_planner_speed_gate"],
        wall_replay_equal=prior["wall_replays_equal"],
        bore_replay_equal=True,
        bore_report_sha256=sha(gauge),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("full", type=Path)
    p.add_argument("second", type=Path)
    p.add_argument("--baseline", type=Path, default=REPORT)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    save_json(a.out, verify(a.full, a.second, a.baseline))
    print("PASS: rebuild, gauges and legacy payloads verified.")
