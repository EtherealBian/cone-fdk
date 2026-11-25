"""Isolated stdlib audit of frozen trial rows. Distinguish invalid evidence from a valid loss."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import review_wall_bench as b  # noqa: E402

REPORT = HERE.parents[1] / "reports"
SOURCE = REPORT / "wall-pack-ab.json"
DEST = REPORT / "spark-20261125/wall-stage-performance"
PIN = "2c4f683740cbe15112bcdd448bc55d615315086527e6fcb51d6f3d83ef181b04"


@pytest.fixture
def receipts():
    return b.read(DEST / "pairs.json", PIN), b.read(SOURCE, b.REFERENCE)


@pytest.mark.parametrize(
    "key",
    [
        "parent",
        "source",
        "tool",
        "adapter",
        "logger",
        "case",
        "levels",
        "budget",
        "omitted",
        "swapped",
        "sequence",
        "zero",
        "negative",
        "float",
        "bool",
        "gpu",
        "zone",
        "wall",
        "status",
        "processes",
        "tiles",
        "launches",
        "read",
        "write",
        "cap",
        "peak",
        "tiny_peak",
    ],
)
def test_bad_evidence_is_not_a_slow_result(receipts, key):
    doc, ref = receipts
    case = doc["cases"]["two-nonnominal"]
    row = case["pairs"][0][0]
    if key == "parent":
        doc["parent_manifest_sha256"] = "0" * 64
    elif key == "source":
        doc["reference_sha256"] = "0" * 64
    elif key == "tool":
        doc["product_sha256"]["wall-pack"] = "f" * 64
    elif key == "adapter":
        doc["adapter_sha256"] = "0" * 64
    elif key == "logger":
        doc["measurement_code_sha256"] = "0" * 64
    elif key == "case":
        del doc["cases"]["eight-level-tiled"]
    elif key == "levels":
        case["levels_per_mm"].reverse()
    elif key == "budget":
        case["budget_mib"] = 1
    elif key == "omitted":
        case["pairs"].pop()
    elif key == "swapped":
        case["pairs"][0].reverse()
    elif key == "sequence":
        row["sequence"] = True
    elif key in ("zero", "negative", "float", "bool"):
        row["elapsed_ns"] = {"zero": 0, "negative": -1, "float": 1.5, "bool": True}[key]
    elif key == "gpu":
        row["gpu"] = "other"
    elif key == "zone":
        row["redzones_verified"] = 1
    elif key == "wall":
        row["level_sha256"].reverse()
    elif key == "status":
        row["status_counts"][0][0] -= 1
    elif key == "tiny_peak":
        row["peak_owned_bytes"] = 1
    else:
        row[
            {
                "processes": "native_processes",
                "tiles": "tiles_per_level",
                "launches": "launches",
                "read": "atlas_bytes",
                "write": "wall_bytes",
                "cap": "budget_bytes",
                "peak": "peak_owned_bytes",
            }[key]
        ] = 0
    with pytest.raises(ValueError):
        b.assess(doc, ref)


@pytest.mark.parametrize(
    "mode,should_pass",
    [
        ("exact_ratio", True),
        ("miss_ratio", False),
        ("four_wins", False),
        ("five_wins", True),
        ("device_more", False),
    ],
)
def test_calculate_gates_not_stored_verdict(receipts, mode, should_pass):
    doc, ref = receipts
    c = doc["cases"]["two-nonnominal"]
    c["passed"] = not should_pass
    for i, pair in enumerate(c["pairs"]):
        row = {r["path"]: r for r in pair}
        row["single"]["elapsed_ns"] = 10000
        row["pack"]["elapsed_ns"] = {
            "exact_ratio": 9500,
            "miss_ratio": 9501,
            "four_wins": 9000 if i < 4 else 11000,
            "five_wins": 9000 if i < 5 else 11000,
            "device_more": 8000,
        }[mode]
        if mode == "device_more":
            row["pack"]["peak_owned_bytes"] = row["single"]["peak_owned_bytes"] + 1
    c["warmup"][0]["elapsed_ns"] = 1
    c["warmup"][1]["elapsed_ns"] = 100000000
    assert b.assess(doc, ref)["all_gates_passed"] is should_pass


def test_cold_cli_external_pin_and_wrong_root():
    cmd = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(HERE / "review_wall_bench.py"),
        str(DEST / "pairs.json"),
        PIN,
        str(SOURCE),
    ]
    p = subprocess.run(cmd, capture_output=True)
    assert p.returncode == 0 and json.loads(p.stdout) == json.loads(
        (DEST / "summary.json").read_bytes()
    )
    cmd[-2] = "0" * 64
    p = subprocess.run(cmd, capture_output=True)
    assert p.returncode == 1 and not p.stdout


def test_measurement_inputs_unmodified_by_reviewer(receipts):
    a, b0 = receipts
    old = copy.deepcopy((a, b0))
    b.assess(a, b0)
    assert (a, b0) == old
