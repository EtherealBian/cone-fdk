"""Alternating measured whole edge-review producers, starting with an already witnessed parent."""

import argparse
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from compare_native import sha
from edge_review import check
from edge_review import run as single
from edge_review_packed import run as packed
from remount_loss import write_doc


def benchmark(parent, work, parent_sha256, companion_sha256):
    work.mkdir(parents=True, exist_ok=False)
    result = dict(
        protocol="One untimed pair followed by six alternating single/packed pairs, fresh child directories. Measured from entry to the published edge manifest, including parent checks, SHA-256, CUDA extraction, copy/unpack, status and reduction. Same resident Python process and warm OS cache. Does not include CT reconstruction, pose estimation or region reporting. Device bytes exclude driver allocations.",
        gate="identical map digests and counts on every trial; >=5% lower median, >=5 of 6 paired wins and no increase in peak native-owned GPU bytes per case",
        parent_sha256=parent_sha256,
        companion_sha256=companion_sha256,
        cases={},
    )
    for case, levels, budget in [
        ("three-level-review", [0.018, 0.020, 0.022], 16),
        (
            "nine-level-review-tiled",
            [0.014, 0.016, 0.018, 0.019, 0.020, 0.022, 0.024, 0.03, 0.04],
            1,
        ),
    ]:
        expected = None

        def trial(variant, levels=levels, budget=budget):
            nonlocal expected
            with TemporaryDirectory(dir=work) as t:
                out = Path(t) / "edge"
                fn, extra = (
                    (single, {})
                    if variant == "single"
                    else (packed, {"companion_sha256": companion_sha256})
                )
                start = time.perf_counter_ns()
                doc = fn(parent, out, levels, parent_sha256=parent_sha256, budget=budget, **extra)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                signature = (
                    doc["outputs_sha256"],
                    doc["counts"],
                    doc["nominal_by_class"],
                    doc["sensitive_missed_by_same_level"],
                )
                check(
                    all(sha(out / n) == s for n, s in signature[0].items()),
                    "published artifact mismatch",
                )
                if expected is None:
                    expected = signature
                check(signature == expected, "semantic or byte mismatch")
                if variant == "single":
                    stages = [
                        s for g in doc["stages"].values() for s in g if s["source"] == "cuda_wall"
                    ]
                    peak = max(s["peak_owned_bytes"] for s in stages)
                    gpu = {s["gpu"] for s in stages}
                    calls = doc["native_wall_calls"]
                else:
                    records = [p["native"] for p in doc["packs"].values()]
                    peak = max(p["peak_owned_bytes"] for p in records)
                    gpu = {p["gpu"] for p in records}
                    calls = doc["native_wall_pack_calls"]
                    check(
                        budget != 1 or all(p["tiles"] > 1 for p in records),
                        "tiling not exercised",
                    )
                check(len(gpu) == 1, "mixed devices")
                return dict(
                    path=variant,
                    elapsed_ms=elapsed,
                    peak_owned_bytes=peak,
                    gpu=gpu.pop(),
                    native_processes=calls,
                )

        trial("single")
        trial("packed")
        rows = [
            [trial(v) for v in (("single", "packed") if i % 2 == 0 else ("packed", "single"))]
            for i in range(6)
        ]
        paired = [{x["path"]: x for x in row} for row in rows]
        times = {
            v: statistics.median(row[v]["elapsed_ms"] for row in paired)
            for v in ("single", "packed")
        }
        wins = sum(row["packed"]["elapsed_ms"] < row["single"]["elapsed_ms"] for row in paired)
        memory_ok = all(
            row["packed"]["peak_owned_bytes"] <= row["single"]["peak_owned_bytes"]
            for row in paired
        )
        check(len({x["gpu"] for row in rows for x in row}) == 1, "changed GPU")
        result["cases"][case] = dict(
            levels_per_mm=levels,
            native_budget_mib=budget,
            median_ms=times,
            paired_wins=wins,
            memory_gate=memory_ok,
            runs=rows,
            matched_outputs_sha256=expected[0],
            counts=expected[1],
            passed=memory_ok and wins >= 5 and times["packed"] <= 0.95 * times["single"],
        )
    write_doc(work / "benchmark.json", result)
    check(
        all(c["passed"] for c in result["cases"].values()),
        "performance gate missed; measurements retained",
    )
    for name, c in result["cases"].items():
        print(name, c["median_ms"], c["paired_wins"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("parent", "work"):
        p.add_argument(k, type=Path)
    for k in ("parent", "companion"):
        p.add_argument("--" + k + "-sha256", required=True)
    benchmark(**vars(p.parse_args()))
