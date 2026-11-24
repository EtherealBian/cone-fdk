"""Independently recount a sealed sensitivity map and bound eligible-pore recall."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pore_byte_radius import DEV, LEVELS, ROOT, SHAPE, array_sha, sha
from pore_patch_stats import stack, verified
from scipy import ndimage as ndi

EVIDENCE = ROOT / "pores/evidence/byte-radius"


def iou_bound(size, hits, minimum=64):
    # The union contains all reference voxels plus the missing minimum object mass.
    if not (size >= minimum >= 1 and 0 <= hits <= size):
        raise ValueError("Invalid population")
    return hits, size + max(0, minimum - hits)


def audit(cache):
    if not __debug__:
        raise RuntimeError("Do not run with -O")
    seal = json.loads((EVIDENCE / "predictions.json").read_text())["specimens"]
    results = json.loads((EVIDENCE / "results.json").read_text())
    summary = {}
    for i in DEV:
        k = str(i)
        r = results[k]
        file = EVIDENCE / f"sample{i}.npz"
        cp = EVIDENCE / f"sample{i}-references.csv"
        assert sha(file) == seal[k]["file"] and sha(cp) == r["references_csv"]
        with np.load(file, allow_pickle=False) as v:
            c = v["signed"]
        assert c.shape == SHAPE and c.dtype == np.int8 and array_sha(c) == seal[k]["signed"]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        truth = stack(seg, i, "segmented", 400, 528)
        # Histograms and per-object crop scans do not call the original bounds scorer.
        h = np.stack(
            [np.bincount(c[truth == t].astype("i2") + 9, minlength=19) for t in (False, True)]
        )
        assert not np.any(h[:, 9])
        assert (
            dict(
                zip(
                    [str(j) for j in range(-9, 10) if j],
                    map(int, h.sum(0)[np.arange(19) != 9]),
                    strict=True,
                )
            )
            == seal[k]["counts"]
        )
        lab, n = ndi.label(truth)
        csvrows = {}
        with cp.open(newline="") as fp:
            for v in csv.DictReader(fp):
                csvrows[int(v["reference_id"])] = {k: int(x) for k, x in v.items()}
        with (ROOT / f"pores/evidence/candidate-h/sample{i}-paired-g-h.csv").open(
            newline=""
        ) as fp:
            previous = {
                int(v["reference_id"]): int(v["reference_voxels"]) for v in csv.DictReader(fp)
            }
        assert previous == {k: v["voxels"] for k, v in csvrows.items()}
        checked = set()
        blocked = {e: [] for e in LEVELS}
        for pid, box in enumerate(ndi.find_objects(lab, max_label=n), 1):
            if any(s.start == 0 or s.stop == d for s, d in zip(box, SHAPE, strict=True)):
                continue
            v = c[box][lab[box] == pid]
            size = len(v)
            if size < 64:
                continue
            assert csvrows[pid]["voxels"] == size
            checked.add(pid)
            for e in LEVELS:
                possible = int(np.count_nonzero(v >= -e))
                assert size - possible == csvrows[pid][f"locked_fn_at_{e}"]
                a, b = iou_bound(size, possible)
                if 2 * a < b:
                    blocked[e].append(
                        dict(reference_id=pid, size=size, max_hits=a, min_union_at_max_hits=b)
                    )
        assert checked == set(csvrows) and len(checked) == r["references"]
        levels = {}
        for e in LEVELS:
            fp = int(h[0, 10 + e :].sum())
            fn = int(h[1, : 9 - e].sum())
            tp = int(h[1].sum()) - fn
            q = r["levels"][str(e)]
            assert [fp, fn] == [q["unavoidable_fp"], q["unavoidable_fn"]]
            assert [2 * tp / (2 * tp + fp + fn), tp / (tp + fp), tp / (tp + fn)] == [
                q[t + "_upper"] for t in ("dice", "precision", "recall")
            ]
            total = len(checked)
            cap = total - len(blocked[e])
            levels[e] = dict(
                reference_count=total,
                match_count_upper=cap,
                reference_coverage_upper=cap / total,
                coverage_gate_impossible=10 * cap < 9 * total,
                impossible_ids=blocked[e],
            )
        summary[i] = levels
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    report = audit(a.cache)
    if a.out:
        with a.out.open("x", encoding="utf8", newline="\n") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
    else:
        assert json.loads(json.dumps(report)) == json.loads(
            (EVIDENCE / "coverage-bounds.json").read_text()
        )
        print("All 15 levels and 112 complete reference pores reproduced.")
