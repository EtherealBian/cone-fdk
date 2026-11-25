"""Bind full GPU acceptance and two independent simulation runs to a reproducible loss build."""

import argparse
import json
from pathlib import Path

from check_bore_regression import verify as verify_legacy
from common import REPORT
from compare_native import sha
from validate_roi import require
from validate_stack import save_json


def sections(root):
    settings = json.loads((root / "bore-gauge.json").read_bytes())["settings"]
    return {
        s: {n: v["sections_sha256"] for n, v in a["cases"].items()} for s, a in settings.items()
    }


def verify(full, replay, baseline):
    old = verify_legacy(full, replay, baseline)
    require(
        old["pytest"]["tests"] >= 195 and sections(full) == sections(baseline),
        "test count or frozen bore payload changed",
    )
    files = (
        "wall-loss.json",
        "wall-loss-example.json",
        "wall-loss-demo.zip",
        "wall-loss-scale.json",
        "wall-loss.png",
    )
    hashes = {n: sha(full / n) for n in files}
    require(
        all(sha(replay / n) == v for n, v in hashes.items()), "independent loss replay changed"
    )
    report = json.loads((full / files[0]).read_bytes())
    build = old["build"]
    require(
        report["binary_sha256"]
        == {"cone-fdk.exe": build["executable_sha256"], "kernels.ptx": build["ptx_sha256"]},
        "not the qualified build",
    )
    require(
        hashes["wall-loss-example.json"] == report["cases"]["dual-surface-0"]["report_sha256"],
        "example was not emitted by the qualified build",
    )
    return dict(
        protocol="Full GPU acceptance; latest tests and loss pack rerun on same binary; fresh wall/bore/loss replay; published legacy payloads pinned.",
        gpu=old["gpu"],
        pytest=old["pytest"],
        build=build,
        loss_replay_sha256=hashes,
        frozen_bore_sections=sections(full),
        legacy=old["legacy"],
        legacy_wall=old["legacy_wall"],
        wall_replay_equal=old["wall_replay_equal"],
        bore_replay_equal=old["bore_replay_equal"],
        planner_gate=old["planner_gate"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("full", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("--baseline", type=Path, default=REPORT)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    save_json(a.out, verify(a.full, a.replay, a.baseline))
    print("PASS: loss inspection, independent replay and frozen legacy outputs")
