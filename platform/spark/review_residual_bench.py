"""Offline audit of a GB10 residual benchmark. No CUDA, third-party modules or filesystem writes."""

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from statistics import median

BUILD = "b83e045b2e66612a48bde9bd92a0864ad26fc0bde08e5417b029478f0da6cd3c"
PROTOCOL = "e4ecf24baa6d6bee9146208a4b8cc587560de1839bfd4120d80212fbd6a8d272"
HARNESS = "345388ac822c22f6b9633c2d0fdf6ab7d1b899d241089485442f89304e02735d"
BANDS = (1, 4, 7, 16)
ORDERS = ((1, 4, 16, 7), (4, 7, 1, 16), (7, 16, 4, 1), (16, 1, 7, 4))
TIMES = ("child_elapsed_ms", "wall_ms", "read_ms", "kernel_ms")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def authenticated(path, pin, limit):
    require(isinstance(pin, str) and re.fullmatch("[0-9a-f]{64}", pin), "invalid SHA-256")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        require(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= limit, "input type/size")
        data = f.read(limit + 1)
    require(
        len(data) == st.st_size and hashlib.sha256(data).hexdigest() == pin,
        "input SHA-256 mismatch",
    )
    return json.loads(data)


def review(q, qp, b, bp):
    require(
        q["format"] == "CFDK-paired-volume-qualification-v1"
        and q["build_sha256"] == BUILD
        and q["protocol_sha256"] == PROTOCOL,
        "unknown qualification",
    )
    require(
        b["format"] == "CFDK-residual-band-benchmark-v1"
        and b["build_sha256"] == BUILD
        and b["harness_sha256"] == HARNESS
        and b["qualified_pair_sha256"] == qp
        and b["source_sha256"] == q["source"]["pins"],
        "benchmark parent or harness mismatch",
    )
    shape = q["shape_xyz"]
    require(
        len(shape) == 3
        and all(type(x) is int and 1 <= x <= 2048 for x in shape)
        and math.prod(shape) <= 134217728,
        "shape",
    )
    nx, ny, nz = shape
    require(
        q["source"]["shape"] == shape
        and q["slices_verified_by_numpy"] == nz
        and q["aggregate"]["voxels"] == nx * ny * nz,
        "qualification workload",
    )
    require(
        b["gpu"] == "NVIDIA GB10"
        and same(b["bands"], list(BANDS))
        and same([b["budget_mib"], b["warmups"], b["timed_samples"]], [64, 4, 32]),
        "treatment design",
    )
    samples = b["samples"]
    require(len(samples) == 32, "sample count")
    by = {t: [] for t in BANDS}
    for index, s in enumerate(samples):
        block, position = divmod(index, 4)
        t = (*ORDERS, *ORDERS[::-1])[block][position]
        require(same([s["block"], s["band"]], [block, t]), "block or order")
        depth = min(nz, t)
        plane = nx * ny
        tiles = (plane + 1023) // 1024
        owned = depth * (8 * plane + 72 * tiles) + 1536
        require(
            same([s["owned_bytes"], s["launches"]], [owned, (nz + depth - 1) // depth])
            and owned <= 64 * 2**20,
            "allocation or launch accounting",
        )
        require(
            all(type(s[k]) in (float, int) and math.isfinite(s[k]) and s[k] >= 0 for k in TIMES),
            "invalid timing",
        )
        require(
            0 < s["child_elapsed_ms"]
            and max(s["read_ms"], s["kernel_ms"]) <= s["wall_ms"] <= s["child_elapsed_ms"],
            "inconsistent clocks",
        )
        by[t].append(s)
    summary = {}
    for t, rows in by.items():
        row = {
            "owned_bytes": rows[0]["owned_bytes"],
            "paired_child_wins_against_band_1": None
            if t == 1
            else sum(
                x["child_elapsed_ms"] < y["child_elapsed_ms"]
                for x, y in zip(rows, by[1], strict=True)
            ),
        }
        for k in TIMES:
            v = [s[k] for s in rows]
            m = median(v)
            row[k] = {
                "min": min(v),
                "max": max(v),
                "median": m,
                "mad": median(abs(x - m) for x in v),
            }
        summary[str(t)] = row
    require(same(summary, b["summary"]), "published statistics differ")
    best = min(BANDS, key=lambda t: summary[str(t)]["child_elapsed_ms"]["median"])
    eligible = [
        t
        for t in BANDS
        if summary[str(t)]["child_elapsed_ms"]["median"]
        <= 1.05 * summary[str(best)]["child_elapsed_ms"]["median"]
    ]
    smallest = min(eligible, key=lambda t: summary[str(t)]["owned_bytes"])
    require(
        same(
            [best, smallest],
            [
                b["minimum_observed_child_median_band"],
                b["smallest_buffer_within_five_percent_of_minimum"],
            ],
        ),
        "selection mismatch",
    )
    return dict(
        format="CFDK-offline-residual-benchmark-review-v1",
        qualification_sha256=qp,
        benchmark_sha256=bp,
        case=q["source"]["case"],
        timed_samples=32,
        best_band=best,
        smallest_near_best_band=smallest,
        summary=summary,
        scope="Authenticated recorded timings and derived statistics; no rerun, storage-cold, physical accuracy or whole-CT speedup claim.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qualification", type=Path)
    p.add_argument("qualification_pin")
    p.add_argument("benchmark", type=Path)
    p.add_argument("benchmark_pin")
    a = p.parse_args()
    try:
        q = authenticated(a.qualification, a.qualification_pin, 2 << 20)
        b = authenticated(a.benchmark, a.benchmark_pin, 128 << 10)
        print(
            json.dumps(
                review(q, a.qualification_pin, b, a.benchmark_pin),
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        OverflowError,
        OSError,
        RecursionError,
    ) as e:
        p.exit(1, f"benchmark review refused: {e}\n")


if __name__ == "__main__":
    main()
