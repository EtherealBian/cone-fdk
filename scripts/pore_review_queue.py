"""Build a label-free disagreement queue from aligned, authenticated prediction runs."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def read(directory, specimen):
    stem = f"sample{specimen}"
    report = json.loads((directory / (stem + ".json")).read_text(encoding="utf8"))
    html = (directory / (stem + "-review.html")).read_bytes()
    if hashlib.sha256(html).hexdigest() != report["review_sha256"]:
        raise ValueError("Review changed")
    p = json.loads(
        re.search(rb'<script id="data" type="application/json">(.*?)</script>', html, re.S)[1]
    )
    if p["specimen"] != specimen:
        raise ValueError("Wrong scan")
    reg = (specimen, report["source"]["raw"]["sha256"], p["shape"], p["origin"], p["spacing"])
    objects = {
        int(k): int(v["row"]["voxels"])
        for k, v in p["objects"].items()
        if int(v["row"]["faces"]) == 0 and int(v["row"]["voxels"]) >= 64
    }
    # The reference layer and fitted thresholds are deliberately never consulted.
    runs = [[seg for seg in layer["objects"] if seg[2] in objects] for layer in p["layers"]]
    return reg, objects, runs


def intersect(left, right):
    """Intersect sorted, disjoint run pairs. IDs may be sparse; no dense volume needed."""
    edges = Counter()
    for a, b in zip(left, right, strict=True):
        i = j = 0
        while i < len(a) and j < len(b):
            x, n, u = a[i]
            y, m, v = b[j]
            end = x + n
            other = y + m
            overlap = min(end, other) - max(x, y)
            if overlap > 0:
                edges[u, v] += overlap
            i += int(end <= other)
            j += int(other <= end)
    return edges


def compare(main, peer):
    if main[0] != peer[0]:
        raise ValueError("Unaligned source, ROI or spacing")
    sizes = main[1]
    other = peer[1]
    best = {}
    for (u, v), n in intersect(main[2], peer[2]).items():
        union = sizes[u] + other[v] - n
        best[u] = max(best.get(u, (0, 0, 0, 0)), (n / union, -v, n, union))
    out = {}
    for u, vol in sizes.items():
        if u not in best:
            out[u] = ("", 0, None, True)
            continue
        q, neg, hit, union = best[u]
        v = -neg
        d = other[v] - vol
        # Use exact integer threshold comparisons including equality, no rounded display values.
        out[u] = (v, q, d, 2 * hit < union or 100 * abs(d) > 15 * vol)
    return out


def queue(main, c, d):
    left = compare(main, c)
    right = compare(main, d)
    rows = []
    for u, vol in main[1].items():
        a, b = left[u], right[u]
        rows.append([u, vol, a[3] or b[3], *a, *b])
    return sorted(rows, key=lambda r: (not r[2], min(r[4], r[8]), r[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate_c", type=Path)
    ap.add_argument("candidate_d", type=Path)
    ap.add_argument("candidate_e", type=Path)
    ap.add_argument("specimens", nargs="+", type=int, choices=(2, 3, 5))
    args = ap.parse_args()
    for i in args.specimens:
        e = read(args.candidate_e, i)
        rows = queue(e, read(args.candidate_c, i), read(args.candidate_d, i))
        with (args.candidate_e / f"sample{i}-review-queue.csv").open(
            "w", encoding="utf8", newline=""
        ) as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "e_pred_id",
                    "e_voxels",
                    "flagged",
                    *[
                        f"{peer}_{key}"
                        for peer in ("c", "d")
                        for key in ("peer_id", "iou", "peer_minus_e_voxels", "disagrees")
                    ],
                ]
            )
            w.writerows(rows)
        print(i, sum(r[2] for r in rows), "flagged of", len(rows))


if __name__ == "__main__":
    main()
