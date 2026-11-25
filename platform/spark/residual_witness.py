"""Recompute one externally pinned candidate/reference slice without NumPy, CUDA or a checkout."""

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
from pathlib import Path

RULES = "e4ecf24baa6d6bee9146208a4b8cc587560de1839bfd4120d80212fbd6a8d272"
FIELDS = [
    "count",
    "equal",
    "over_2^-20",
    "over_2^-18",
    "over_2^-16",
    "over_2^-14",
    "max_abs",
    "first_yx_flat",
    "signed_sum",
    "abs_sum",
    "sse",
    "reference_energy",
]
CUTS = tuple(2.0**i for i in (-20, -18, -16, -14))


def need(value, why):
    if not value:
        raise ValueError(why)


def pin(p):
    need(isinstance(p, str) and re.fullmatch("[a-f0-9]{64}", p), "expected external SHA-256")
    return p


def hashed(b):
    return hashlib.sha256(b).hexdigest()


def bounded(path, limit, expected, exact=None):
    pin(expected)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(fd)
        need(
            stat.S_ISREG(st.st_mode)
            and 0 < st.st_size <= limit
            and (exact is None or st.st_size == exact),
            "input type or length",
        )
        data = f.read(st.st_size)
        need(
            len(data) == st.st_size and not f.read(1) and hashed(data) == expected,
            "input digest or length changed",
        )
        return data


def agree(actual, oracle):
    need(type(actual) is list and len(actual) == len(FIELDS) == len(oracle), "row format")
    for i in (0, 1, 2, 3, 4, 5, 7):
        need(
            type(actual[i]) is int and actual[i] == oracle[i],
            "different count or first maximum: " + FIELDS[i],
        )
    need(type(actual[6]) in (int, float) and actual[6] == oracle[6], "different maximum")
    for i in range(8, 12):
        scale = oracle[9] if i == 8 else oracle[i]
        need(
            type(actual[i]) in (int, float)
            and math.isfinite(actual[i])
            and abs(actual[i] - oracle[i]) <= 2e-13 * scale,
            "different " + FIELDS[i],
        )


def scalar(a, b):
    need(len(a) > 0 and len(a) == len(b) and len(a) % 4 == 0, "slice pair length")
    count = len(a) // 4

    # One bounded pass at a time so the portable reviewer doesn't retain per-voxel Python objects.
    def pairs():
        return (
            (x[0], y[0])
            for x, y in zip(struct.iter_unpack("<f", a), struct.iter_unpack("<f", b), strict=True)
        )

    counts = [0] * 5
    peak = -1.0
    first = None
    for i, (x, y) in enumerate(pairs()):
        need(math.isfinite(x) and math.isfinite(y), "nonfinite volume")
        v = abs(x - y)
        counts[0] += x == y
        for k, c in enumerate(CUTS, 1):
            counts[k] += v > c
        if v > peak:
            first, peak = i, v
    return [
        count,
        *counts,
        peak,
        first,
        math.fsum(x - y for x, y in pairs()),
        math.fsum(abs(x - y) for x, y in pairs()),
        math.fsum((x - y) * (x - y) for x, y in pairs()),
        math.fsum(y * y for _, y in pairs()),
    ]


def verify(report_path, expected, z, candidate, reference):
    doc = json.loads(bounded(report_path, 8 << 20, expected))
    need(
        doc["format"] == "CFDK-paired-volume-qualification-v1"
        and doc["protocol_sha256"] == RULES
        and doc["row_fields"] == FIELDS,
        "unrecognized audit",
    )
    nx, ny, nz = doc["shape_xyz"]
    need(
        all(type(i) is int and 1 <= i <= 2048 for i in (nx, ny, nz))
        and nx * ny * nz <= 134217728
        and type(z) is int
        and 0 <= z < nz
        and len(doc["planes"]) == nz,
        "invalid grid or z",
    )
    row = doc["planes"][z]
    need(row["z"] == z, "wrong plane binding")
    size = 4 * nx * ny
    a = bounded(candidate, size, row["candidate_sha256"], size)
    b = bounded(reference, size, row["reference_sha256"], size)
    stats = scalar(a, b)
    agree(row["gpu"], stats)
    i = stats[7]
    return dict(
        format="CFDK-paired-local-witness-v1",
        qualification_sha256=expected,
        z=z,
        voxels=nx * ny,
        maximum_xyz=[i % nx, i // nx, z],
        max_abs=stats[6],
        equal=stats[1],
        strict_exceedance=stats[2:6],
        scope="Only this authenticated slice was independently remeasured.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("report_sha256")
    p.add_argument("z", type=int)
    p.add_argument("candidate", type=Path)
    p.add_argument("reference", type=Path)
    a = p.parse_args()
    try:
        print(
            json.dumps(
                verify(a.report, a.report_sha256, a.z, a.candidate, a.reference), allow_nan=False
            )
        )
    except (ValueError, OSError, KeyError, TypeError, IndexError) as e:
        p.exit(1, f"witness refused: {e}\n")
