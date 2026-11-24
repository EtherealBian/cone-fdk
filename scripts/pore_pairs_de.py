"""Compare two methods by reference component, including pores missed by both."""

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def records(directory, i, suffix):
    with (directory / f"sample{i}-{suffix}.csv").open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


def load(directory, i):
    report = json.loads((directory / f"sample{i}.json").read_text(encoding="utf8"))
    matches = {int(r["ref_scipy_label"]): r for r in records(directory, i, "matches")}
    refs = {}
    best = {}
    intersection = {}
    for r in records(directory, i, "overlap"):
        if r["ref_eligible"] != "True":
            continue
        ref = int(r["ref_label"])
        pred = int(r["pred_id"])
        v = int(r["ref_voxels"])
        p = int(r["pred_voxels"])
        n = int(r["intersection_voxels"])
        if refs.setdefault(ref, v) != v:
            raise ValueError("Reference size changed within a scan")
        intersection[ref, pred] = n
        # For a miss, the best single IoU helps distinguish no detection, size rejection,
        # heavy merging, and an insufficient IoU. Never replace the formal match with it.
        q = n / (v + p - n)
        best[ref] = max(best.get(ref, (-1, 0, 0, "")), (q, pred, p, r["pred_eligible"]))
    if (
        len(refs) != report["objects"]["eligible_reference"]
        or len(matches) != report["objects"]["matches"]
        or matches.keys() - refs.keys()
    ):
        raise ValueError("Incomplete component tables")
    return refs, matches, best, intersection


def descriptor(r, ref, vol):
    _, matches, best, intersection = r
    if ref not in matches:
        q, p, n, eligible = best[ref]
        return ["", "", "", "", "", p or "", n, eligible, q]
    row = matches[ref]
    p = int(row["pred_id"])
    n = int(row["pred_voxels"])
    hit = intersection[ref, p]
    delta = n - vol
    if abs((100 * delta / vol) - float(row["signed_volume_error_pct"])) > 1e-9:
        raise ValueError("Signed-volume mismatch")
    return [p, n, delta, vol - hit, n - hit, "", "", "", ""]


def paired(a, b):
    if a[0] != b[0]:
        raise ValueError("Reference IDs or sizes differ between methods")
    result = []
    for ref, vol in sorted(a[0].items()):
        left = descriptor(a, ref, vol)
        right = descriptor(b, ref, vol)
        label = (
            ("shared" if left[0] else "gained")
            if right[0]
            else ("lost" if left[0] else "neither")
        )
        result.append([ref, vol, label, *left, *right])
    return result


def output(rows):
    stream = io.StringIO(newline="")
    w = csv.writer(stream, lineterminator="\r\n")
    cols = [
        "pred_id",
        "pred_voxels",
        "signed_delta_vox",
        "reference_only_vox",
        "prediction_only_vox",
        "miss_best_iou_pred_id",
        "miss_best_iou_pred_vox",
        "miss_best_pred_eligible",
        "miss_best_iou",
    ]
    w.writerow(
        [
            "reference_id",
            "reference_voxels",
            "status",
            *[f"{prefix}_{c}" for prefix in ("d", "e") for c in cols],
        ]
    )
    w.writerows(rows)
    return stream.getvalue().encode("utf8")


def summarize(rows):
    def p95(arr):
        return f"{np.percentile(arr, 95):.2f}%" if len(arr) else "—"

    shared = [r for r in rows if r[2] == "shared"]
    dv = np.array([abs(r[5]) * 100 / r[1] for r in shared])
    ev = np.array([abs(r[14]) * 100 / r[1] for r in shared])
    small = [r for r in rows if 64 <= r[1] < 128]
    sm = [r for r in small if r[2] == "shared"]

    def ids(status):
        return ", ".join(str(r[0]) for r in rows if r[2] == status) or "—"

    return [
        len(shared),
        p95(dv),
        p95(ev),
        f"{sum(ev < dv)} / {sum(ev == dv)} / {sum(ev > dv)}",
        ids("gained"),
        ids("lost"),
        ids("neither"),
    ], [
        len(small),
        sum(r[2] in ("shared", "lost") for r in small),
        sum(r[2] in ("shared", "gained") for r in small),
        len(sm),
        p95([abs(r[5]) * 100 / r[1] for r in sm]),
        p95([abs(r[14]) * 100 / r[1] for r in sm]),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for i in (2, 3, 5):
        rows = paired(load(args.old, i), load(args.new, i))
        path = args.new / f"sample{i}-paired-d-e.csv"
        text = output(rows)
        if args.check:
            if path.read_bytes() != text:
                raise ValueError(f"Stale paired ledger: {path}")
        else:
            path.write_bytes(text)
        print(i, *summarize(rows), sep=" | ")


if __name__ == "__main__":
    main()
