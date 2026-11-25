"""Independently audit a pinned GB10 wall-stage benchmark and rederive paired gates; stdlib."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

REFERENCE = "c9aea1dcb7580bcbad3412416d11704ee33f86c52fab59a5865261fc5848b601"
PRODUCTS = {
    "cone-fdk": "144a9d873a6f3047637303dff2cdfe4dbe2cb1890b1d8945b1439b5d5da60883",
    "wall-pack": "6351b4db3aa52e1df6a9651ccc329dcb238c45c6cdbc498dbd0c6cef9a72e7dc",
    "kernels.ptx": "2ff008d243a421c97c93f6c94031484d9ec165084855173386e6a491dce2d167",
}
EXPECTED = ("two-nonnominal", "eight-level-tiled")


def need(v, why):
    if not v:
        raise ValueError(why)


def read(p, pin):
    with p.open("rb") as f:
        b = f.read((4 << 20) + 1)
    need(len(b) <= 4 << 20 and hashlib.sha256(b).hexdigest() == pin, "input SHA-256 or size")
    return json.loads(b)


def assess(doc, reference):
    need(
        doc["format"] == "CFDK-GB10-wall-stage-pairs-v1"
        and doc["reference_sha256"] == REFERENCE
        and doc["product_sha256"] == PRODUCTS
        and doc["measurement_code_sha256"]
        == "7c4a7080fe77d325697b4e1c828bc36b8a83972f036630510fc9ad70cc44e5f2"
        and doc["adapter_sha256"]
        == "22d215d0cf481b31972393f9c633431673aa04226385dcb21622e4516c94e28d",
        "wrong provenance",
    )
    need(
        all(
            doc[k] == reference[k]
            for k in ("parent_manifest_sha256", "atlas_sha256", "grid_sha256")
        ),
        "wrong case inputs",
    )
    need(set(doc["cases"]) == set(reference["cases"]) == set(EXPECTED), "case set")
    out = {}
    for name in EXPECTED:
        ref = reference["cases"][name]
        case = doc["cases"][name]
        levels = ref["levels_per_mm"]
        k = len(levels)
        budget = ref["budget_mib"]
        tiles = 1 if k == 2 else 8
        n = 14760
        atlas_bytes = 7143840
        need(case["levels_per_mm"] == levels and case["budget_mib"] == budget, "case protocol")
        rows = [case["warmup"], *case["pairs"]]
        need(len(rows) == 7, "six measured pairs required")
        elapsed = {"single": [], "pack": []}
        wins = 0
        memory = True
        for i, pair in enumerate(rows):
            order = ("pack", "single") if i > 0 and i % 2 == 0 else ("single", "pack")
            need(len(pair) == 2 and tuple(r["path"] for r in pair) == order, "wrong paired order")
            by = {r["path"]: r for r in pair}
            for kind, r in by.items():
                seq = i * 2 + order.index(kind)
                need(type(r["sequence"]) is int and r["sequence"] == seq, "missing trial")
                p = kind == "pack"
                t = r["elapsed_ns"]
                need(type(t) is int and 0 < t < 600 * 10**9, "elapsed_ns")
                need(
                    r["gpu"] == "NVIDIA GB10"
                    and r["redzones_verified"] is True
                    and r["level_sha256"] == ref["level_sha256"]
                    and r["status_counts"] == ref["status_counts"],
                    "numerical evidence",
                )
                need(
                    all(
                        len(c) == 7 and all(type(v) is int and v >= 0 for v in c) and sum(c) == n
                        for c in r["status_counts"]
                    ),
                    "status counts",
                )
                expected = {
                    "native_processes": 1 if p else k,
                    "tiles_per_level": tiles,
                    "launches": tiles * k,
                    "atlas_bytes": atlas_bytes * (1 if p else k),
                    "wall_bytes": 32 * n * k,
                    "budget_bytes": budget << 20,
                }
                need(
                    all(type(r[x]) is int and r[x] == v for x, v in expected.items()),
                    "native accounting",
                )
                need(
                    type(r["peak_owned_bytes"]) is int
                    and (32 + 121 * 4) * ((n + tiles - 1) // tiles)
                    <= r["peak_owned_bytes"]
                    <= budget << 20,
                    "peak device bytes",
                )
                if i:
                    elapsed[kind].append(t)
            if i:
                wins += by["pack"]["elapsed_ns"] < by["single"]["elapsed_ns"]
                memory &= by["pack"]["peak_owned_bytes"] <= by["single"]["peak_owned_bytes"]
        med = {kind: statistics.median(ts) / 1e6 for kind, ts in elapsed.items()}
        ratio = med["single"] / med["pack"]
        need(math.isfinite(ratio), "nonfinite ratio")
        out[name] = dict(
            measured_pairs=6,
            median_ms=med,
            process_wall_time_ratio=ratio,
            paired_wins=wins,
            peak_device_memory_gate=memory,
            atlas_read_reduction_factor=k,
            passed=bool(memory and wins >= 5 and med["pack"] <= 0.95 * med["single"]),
        )
    return dict(
        cases=out,
        all_gates_passed=all(x["passed"] for x in out.values()),
        scope="Warm-cache per-path process + materialization wall time; no scanner, CT reconstruction, snapshotting or end-to-end speed claim.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("sha256")
    p.add_argument("reference", type=Path)
    a = p.parse_args()
    try:
        result = assess(read(a.receipt, a.sha256), read(a.reference, REFERENCE))
    except (OSError, KeyError, TypeError, ValueError, IndexError) as err:
        p.exit(1, f"wall benchmark refused: {err}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["all_gates_passed"] else 2)
