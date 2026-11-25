"""Bind a fresh full run, archived witness parents and six-pair CUDA timings before release."""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median

from compare_native import sha
from remount_loss import write_doc
from validate_roi import require


def timing(bench, q, gpu):
    c = q["cases"]["noisy-change"]
    require(
        bench["parent_sha256"] == c["parent_sha256"]
        and bench["companion_sha256"] == q["qualified_build"]["executable_sha256"],
        "timing provenance",
    )
    groups = bench["cases"]
    require(set(groups) == {"three-level-review", "nine-level-review-tiled"}, "timing cases")
    require(
        groups["three-level-review"]["matched_outputs_sha256"] == c["matched_outputs_sha256"],
        "timed artifacts differ from qualified maps",
    )
    results = {}
    for name, k, budget in (("three-level-review", 3, 16), ("nine-level-review-tiled", 9, 1)):
        g = groups[name]
        rows = g["runs"]
        require(
            len(rows) == 6
            and len(set(g["levels_per_mm"])) == k
            and g["native_budget_mib"] == budget,
            "wrong timing design",
        )
        durations = {p: [] for p in ("single", "packed")}
        wins = 0
        for i, row in enumerate(rows):
            require(
                [v["path"] for v in row]
                == (["single", "packed"] if i % 2 == 0 else ["packed", "single"]),
                "trial order",
            )
            pair = {v["path"]: v for v in row}
            for p, x in pair.items():
                t = x["elapsed_ms"]
                require(
                    math.isfinite(t)
                    and t > 0
                    and x["gpu"] == gpu
                    and x["native_processes"] == (2 * (k - 1) if p == "single" else 2),
                    "bad trial",
                )
                durations[p].append(t)
            require(
                0
                < pair["packed"]["peak_owned_bytes"]
                <= pair["single"]["peak_owned_bytes"]
                <= budget << 20,
                "device budget regression",
            )
            wins += pair["packed"]["elapsed_ms"] < pair["single"]["elapsed_ms"]
        mid = {p: median(ts) for p, ts in durations.items()}
        require(
            mid == g["median_ms"]
            and wins == g["paired_wins"]
            and wins >= 5
            and mid["packed"] <= 0.95 * mid["single"]
            and g["memory_gate"]
            and g["passed"],
            "timing gate",
        )
        results[name] = dict(median_ms=mid, paired_wins=wins)
    return results


def verify(full, archived, benchmark, regression, specimen):
    def one(pattern):
        matches = list(full.glob(pattern))
        require(len(matches) == 1, "require a fresh, single-run report directory: " + pattern)
        return matches[0]

    fresh = one("packed-edge-replay-*.json")
    second = one("specimen-replay-*.json")
    require(
        fresh.read_bytes() == archived.read_bytes()
        and second.read_bytes() == specimen.read_bytes(),
        "fresh and archived replay differ",
    )
    q = json.loads(fresh.read_bytes())
    old = json.loads(regression.read_bytes())
    tests = old["pytest"]
    require(
        tests["tests"] >= 400 and not any(tests[k] for k in ("errors", "skipped", "failures")),
        "full tests",
    )

    def suite(path):
        attrib = ET.parse(path).find("testsuite").attrib
        return {k: int(attrib[k]) for k in ("tests", "errors", "skipped", "failures")}

    require(suite(full / "native-tests.xml") == tests, "full-suite record mismatch")
    final = (
        full / "native-tests.xml" if tests["tests"] >= 417 else full / "native-tests-final.xml"
    )
    final_tests = suite(final)
    require(
        final_tests["tests"] >= 417
        and not any(final_tests[k] for k in ("errors", "skipped", "failures")),
        "portable sample and release tests",
    )
    require(
        q["qualified_build"] == json.loads((full / "wall-pack-build.json").read_bytes())
        and old["build"]["ptx_sha256"] == q["qualified_build"]["ptx_sha256"],
        "changed native build",
    )
    require(
        all(sha(full / n) == s for n, s in q["trusted_records_sha256"].items()),
        "old report not reproduced",
    )
    scores = timing(json.loads(benchmark.read_bytes()), q, old["gpu"])
    return dict(
        protocol="Fresh clean Git checkout of the qualification inputs, repeat native build and full non-CoreOnly acceptance with both optional campaign gates. Archived witness replay versus newly projected witness replay must agree byte-for-byte. Recalculate timing gates from individual recorded trials.",
        gpu=old["gpu"],
        full_pipeline_pytest=tests,
        final_pytest=final_tests,
        final_junit_sha256=sha(final),
        full_region_regression_sha256=sha(regression),
        fresh_and_archived_packed_qualification_sha256=sha(fresh),
        fresh_and_archived_specimen_replay_sha256=sha(second),
        benchmark_sha256=sha(benchmark),
        timing=scores,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("full", "archived", "benchmark", "regression", "specimen", "output"):
        p.add_argument(n, type=Path)
    a = p.parse_args()
    write_doc(a.output, verify(a.full, a.archived, a.benchmark, a.regression, a.specimen))
    print(
        "PASS: source checkout, rebuilt pipelines, archived parents, recorded AB/BA and full regression"
    )
