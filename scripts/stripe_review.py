"""Summarize a pinned wall-review one axial band at a time; no dense labels or fake coverage."""

import argparse
import hashlib
import json
import math
import os
from contextlib import ExitStack
from itertools import product
from pathlib import Path

import numpy as np
from anchor_audit import pin, publish
from edge_review import CLASSES, MAPS, ROLES, check, scenarios, thickness
from region_review import grid_header, read_bound
from stripe_frontier import Frontier
from wall import WALL_DTYPE

MAX_BAND_CELLS = 65536
MAX_PAIR_CELLS = 512 << 20
MAX_FRONTIER_SLOTS = 1 << 20


def scan(job, doc, digest, anchors, band_rows):
    hashes = doc["outputs_sha256"]
    header, _ = read_bound(job / "material.grid", 4096, pin(hashes["material.grid"]))
    (h, w), lo, hi, _ = grid_header(header)
    levels = scenarios(doc["levels_per_mm"], doc["nominal_level_per_mm"])
    k = len(levels)
    n = h * w
    idx = levels.index(doc["nominal_level_per_mm"])
    gate = doc["loss_gate_mm"]
    check(
        doc["class_codes"] == list(CLASSES)
        and levels == doc["levels_per_mm"]
        and doc["shape_ha"] == [h, w]
        and math.isfinite(hi)
        and hi > lo
        and doc["independent_pairs"] == k * k
        and type(gate) in (int, float)
        and math.isfinite(gate)
        and gate > 0,
        "bad review metadata",
    )
    check(
        type(band_rows) is int
        and 1 <= band_rows <= 4096
        and w * min(band_rows, h) <= MAX_BAND_CELLS
        and n * k * k <= MAX_PAIR_CELLS
        and 2 * k * k * w <= MAX_FRONTIER_SLOTS,
        "stripe budget exceeded",
    )
    check(
        len(anchors) == 2
        and anchors[0] != anchors[1]
        and all(
            len(p) == 2 and all(type(v) is int for v in p) and 0 <= p[0] < h and 0 <= p[1] < w
            for p in anchors
        ),
        "two distinct anchors inside material grid required",
    )
    pairs = list(product(range(k), repeat=2))
    measured = [Frontier(w, anchors) for _ in pairs]
    possible = [Frontier(w, anchors) for _ in pairs]
    unknown = [0] * len(pairs)
    endpoints = [[None, None] for _ in pairs]
    table = np.zeros((3, 4), "i8")
    missed = 0
    names = [[f"{r}-{j:02}.wall" for j in range(k)] for r in ROLES]
    sizes = {s: 32 for group in names for s in group}
    sizes.update(dict(zip(MAPS, (1, 8, 8), strict=True)))
    digests = {s: hashlib.sha256() for s in sizes}
    read_bytes = 0
    bands = 0
    with ExitStack() as ctx:
        handles = {s: ctx.enter_context((job / s).open("rb")) for s in sizes}
        check(
            all(
                os.fstat(f.fileno()).st_size == n * sizes[s] and pin(hashes[s])
                for s, f in handles.items()
            ),
            "invalid artifact size or identity",
        )

        def read(s, count, dtype):
            nonlocal read_bytes
            b = handles[s].read(count * sizes[s])
            check(len(b) == count * sizes[s], "short payload: " + s)
            digests[s].update(b)
            read_bytes += len(b)
            return np.frombuffer(b, dtype)

        for y in range(0, h, band_rows):
            rows = min(band_rows, h - y)
            count = rows * w
            bands += 1
            # Only float64 thickness survives per level; the raw 32-byte records are validated then dropped.
            ref, part = [
                [
                    thickness(read(s, count, WALL_DTYPE), lo, hi).reshape(rows, w).copy()
                    for s in group
                ]
                for group in names
            ]
            begin = np.full((rows, w), np.inf)
            end = -begin
            diag_any = np.zeros((rows, w), bool)
            diag_all = np.ones((rows, w), bool)
            nom = None
            for z, (i, j) in enumerate(pairs):
                delta = ref[i] - part[j]
                hit = delta >= gate
                blind = np.isnan(delta)
                unknown[z] += int(blind.sum())
                measured[z].feed(hit)
                possible[z].feed(hit | blind)
                for t, (yy, x) in enumerate(anchors):
                    if y <= yy < y + rows:
                        endpoints[z][t] = (
                            "unknown"
                            if blind[yy - y, x]
                            else "flagged"
                            if hit[yy - y, x]
                            else "below"
                        )
                np.minimum(begin, delta, out=begin)
                np.maximum(end, delta, out=end)
                if i == j:
                    diag_any |= hit
                    diag_all &= hit
                    if i == idx:
                        nom = np.where(blind, 2, hit.astype("u1"))
            code = np.select(
                [np.isnan(begin), begin >= gate, end >= gate], [3, 2, 1], default=0
            ).astype("u1")
            missed += int(((code == 1) & (diag_all == diag_any)).sum())
            table += np.bincount((4 * nom + code).ravel(), minlength=12).reshape(3, 4)
            for s, dtype, expected in zip(
                MAPS, ("u1", "<f8", "<f8"), (code, begin, end), strict=True
            ):
                check(
                    np.array_equal(
                        read(s, count, dtype).reshape(rows, w), expected, equal_nan=True
                    ),
                    "explicit pairs differ from stored envelope: " + s,
                )
        for s, f in handles.items():
            check(
                not f.read(1) and digests[s].hexdigest() == hashes[s],
                "artifact SHA-256 or final size differs: " + s,
            )
    check(
        dict(zip(CLASSES, table.sum(0).tolist(), strict=True)) == doc["counts"]
        and {
            key: dict(zip(CLASSES, row.tolist(), strict=True))
            for key, row in zip(("below", "flagged", "unknown"), table, strict=True)
        }
        == doc["nominal_by_class"]
        and missed == doc["sensitive_missed_by_same_level"],
        "parent class census mismatch",
    )
    result = []
    for z, p in enumerate(pairs):
        m = measured[z].result()
        opt = possible[z].result()["anchors_linked"]
        check(all(endpoints[z]) and (not m["anchors_linked"] or opt), "anchor reconciliation")
        state = (
            "anchor_below_gate"
            if "below" in endpoints[z]
            else "observed_link"
            if m["anchors_linked"]
            else "unresolved_blind_path"
            if opt
            else "separated_on_grid"
        )
        result.append(
            dict(
                pair=list(p),
                observed_components=m["components"],
                largest_observed_cells=m["largest_cells"],
                observed_flagged_cells=m["flagged_cells"],
                unknown_cells=unknown[z],
                anchor_cells=endpoints[z],
                anchor_state=state,
            )
        )
    summary = dict(
        shape_ha=[h, w],
        anchors_row_col=anchors,
        levels_per_mm=levels,
        nominal_pair=[idx, idx],
        loss_gate_mm=gate,
        edge_manifest_sha256=digest,
        scenarios=result,
    )
    return summary, dict(
        bands=bands,
        band_rows=band_rows,
        peak_band_cells=min(band_rows, h) * w,
        max_retained_components_per_frontier=max(f.max_active for f in measured + possible),
        payload_bytes_read=read_bytes,
        hashed_payloads=len(handles),
        pair_cells=n * k * k,
    )


def review(job, expected, anchors, band_rows):
    raw, digest = read_bound(job / "manifest.json", 1 << 20, pin(expected))
    doc = json.loads(raw)
    check(doc["format"] == "CFDK-edge-review-v1", "not an edge review")
    summary, stats = scan(job, doc, digest, anchors, band_rows)
    return dict(
        format="CFDK-striped-region-summary-v1",
        scope="Observed counts and two-anchor connectivity on the sampled cylinder only; no dense labels, peak values or per-component areas are exported.",
        result=summary,
        work=stats,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("edge", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--edge-sha256", required=True)
    p.add_argument("--band-rows", type=int, default=16)
    for s in ("start", "end"):
        p.add_argument("--" + s, nargs=2, type=int, required=True, metavar=("ROW", "COL"))
    a = p.parse_args()
    try:
        d = review(a.edge, a.edge_sha256, [a.start, a.end], a.band_rows)
        publish(a.out, d)
    except (OSError, ValueError, KeyError, TypeError) as err:
        p.exit(1, f"stripe audit failed: {err}\n")
    print(json.dumps(d["work"]))
