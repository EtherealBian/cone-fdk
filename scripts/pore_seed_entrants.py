"""Post-score diagnosis: what reference support overlaps each newly eligible prediction?"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, Z, digest, sha, stack, verified, write
from pore_axial_score import read_mask
from pore_small_seed import ARMS, E
from pore_transfer import label


def audit(cache, score, out):
    seal = json.loads((E / "predictions.json").read_bytes())
    res = {}
    for i in DEV:
        record = json.loads((score / f"sample{i}.json").read_bytes())
        r = seal["specimens"][str(i)]
        assert record["pipeline_id"] == seal["pipeline_id"]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        t = stack(seg, i, "segmented", *Z)
        assert digest(t) == record["reference_mask"]
        lr, rs, inside, _ = label(t)
        arms = {}
        for k in ARMS:
            m = read_mask(E, i, k, r["masks"][k])
            lp, ns, whole, boxes = label(m)
            rows = []
            for e in record["comparisons"][k]["entrants"]:
                pid = lp.flat[e["final_id"] - 1]
                box = boxes[pid - 1]
                sel = lp[box] == pid
                assert whole[pid] and ns[pid] == e["final"] == sel.sum()
                ov = np.bincount(lr[box][sel], minlength=len(rs))
                assert ov.sum() == e["final"]
                ov[0] = 0
                # Max intersection; ties use smallest scan-order reference label.
                j = int(ov.argmax())
                hit = int(ov[j])
                size = int(rs[j]) if hit else 0
                cat = (
                    "no overlap"
                    if not hit
                    else "face"
                    if not inside[j]
                    else "below 64"
                    if size < 64
                    else "eligible"
                )
                rows.append(
                    dict(
                        seed=e["original_seed"],
                        prediction=e["final_id"],
                        voxels=e["final"],
                        matched=e["matched_reference"],
                        largest_ref=j if hit else None,
                        ref_voxels=size,
                        intersection=hit,
                        iou=hit / (size + e["final"] - hit),
                        category=cat,
                    )
                )
            arms[k] = dict(
                rows=rows,
                unmatched_counts=dict(
                    Counter(x["category"] for x in rows if x["matched"] is None)
                ),
            )
        print(i, {k: v["unmatched_counts"] for k, v in arms.items()}, flush=True)
        res[str(i)] = dict(
            reference_mask=digest(t), score_sha256=sha(score / f"sample{i}.json"), arms=arms
        )
    write(
        out,
        dict(
            protocol="Post-score descriptive analysis; maximum non-background intersection, tie by smallest label. Not relabeling or a classifier.",
            specimens=res,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    for s in ("cache", "score", "out"):
        ap.add_argument("--" + s, type=Path, required=True)
    a = ap.parse_args()
    audit(a.cache, a.score, a.out)
