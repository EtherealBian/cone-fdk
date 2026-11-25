"""Recompute the archived cylindrical footprint label grids with the Python standard library.

Input masks are derived classifications (below/observed/unknown), not raw scans.
Labels use row-major cell + 1, wrap only angular columns, and never bridge an unknown cell.
"""

import argparse
import hashlib
import json
import re
import struct
import zlib
from pathlib import Path

FORMAT = "CFDK-GB10-directory-edge-qualification-v1"
CASES = {
    "repeat": [0.018, 0.020, 0.022],
    "noisy-change": [0.018, 0.020, 0.022],
    "invalid-edge-control": [0.020, 0.040],
}
SHAPE = [41, 360]


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def bounded(path, limit):
    with Path(path).open("rb") as f:
        b = f.read(limit + 1)
    need(0 < len(b) <= limit, "file outside review size limit")
    return b


def label(mask, h, w):
    need(h >= 1 and w >= 3 and h * w == len(mask) and len(mask) <= 1 << 20, "invalid grid")
    need(not (set(mask) - {0, 1, 2}), "unknown mask code")
    grid = [-1 if c == 2 else 0 for c in mask]
    regions = 0
    for i, c in enumerate(mask):
        if c != 1 or grid[i]:
            continue
        regions += 1
        ident = i + 1
        grid[i] = ident
        todo = [i]
        while todo:
            u = todo.pop()
            r, x = divmod(u, w)
            for v in (r * w + (x - 1) % w, r * w + (x + 1) % w, u - w, u + w):
                if 0 <= v < len(mask) and mask[v] == 1 and not grid[v]:
                    grid[v] = ident
                    todo.append(v)
    raw = struct.pack(f"<{len(grid)}i", *grid)
    return hashlib.sha256(raw).hexdigest(), regions, mask.count(2), mask.count(1)


def review(receipt, expected, pack):
    data = bounded(receipt, 1 << 20)
    need(
        isinstance(expected, str)
        and re.fullmatch("[0-9a-f]{64}", expected)
        and hashlib.sha256(data).hexdigest() == expected,
        "wrong receipt digest",
    )
    doc = json.loads(data)
    need(doc["format"] == FORMAT and set(doc["cases"]) == set(CASES), "wrong case set")
    meta = doc["offline_masks"]
    compressed = bounded(pack, 1 << 20)
    need(
        meta["format"] == "CFDK-mask-u8-zlib-v1"
        and meta["case_order"] == list(CASES)
        and hashlib.sha256(compressed).hexdigest() == meta["sha256"],
        "wrong mask package",
    )
    h, w = SHAPE
    n = h * w
    size = n * sum(len(x) ** 2 for x in CASES.values())
    codec = zlib.decompressobj()
    raw = codec.decompress(compressed, size + 1)
    need(
        len(raw) == size == meta["uncompressed_bytes"]
        and codec.eof
        and not codec.unused_data
        and not codec.unconsumed_tail,
        "invalid mask stream",
    )
    offset = 0
    counts = {}
    for k, levels in CASES.items():
        case = doc["cases"][k]
        pairs = [[i, j] for i in range(len(levels)) for j in range(len(levels))]
        m = len(pairs)
        need(
            case["shape_ha"] == SHAPE
            and case["levels_per_mm"] == levels
            and case["pairs"] == pairs,
            "wrong scenario grid",
        )
        fields = (
            "periodic_label_sha256",
            "region_counts",
            "unknown_cells",
            "observed_flagged_cells",
        )
        need(all(len(case[f]) == m for f in fields), "incomplete scenario table")
        for i in range(m):
            item = label(raw[offset : offset + n], h, w)
            offset += n
            need(item == tuple(case[f][i] for f in fields), f"{k} pair {pairs[i]} differs")
        counts[k] = m
    return {
        "scenario_grids": counts,
        "cells_checked": offset,
        "scope": "classified synthetic masks only",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("receipt_sha256")
    p.add_argument("masks", type=Path)
    a = p.parse_args()
    print(json.dumps(review(a.receipt, a.receipt_sha256, a.masks)))
