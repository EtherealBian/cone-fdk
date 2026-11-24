"""Score both public section-screen arms on the same interval and reference IDs."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_section_predict import (
    DEV,
    ROOT,
    SHAPE,
    Z,
    digest,
    identity,
    sha,
    stack,
    token,
    verified,
)
from pore_transfer import label, measure

SEAL = ROOT / "pores/evidence/section-area/predictions.json"
ARMS = ("control", "area")


def compare(b, a):
    rr, ss = b["references"], a["references"]
    if rr.keys() != ss.keys() or any(
        rr[k]["reference_voxels"] != ss[k]["reference_voxels"] for k in rr
    ):
        raise ValueError("Mismatched references")
    ids = [k for k in rr if rr[k]["voxels"] is not None and ss[k]["voxels"] is not None]
    tails = [
        float(
            np.percentile(
                [
                    100
                    * abs(r[k]["voxels"] - r[k]["reference_voxels"])
                    / r[k]["reference_voxels"]
                    for k in ids
                ],
                95,
            )
        )
        if ids
        else None
        for r in (rr, ss)
    ]
    criteria = {
        "support": len(ids) >= 5,
        "p95_gain": bool(ids) and tails[0] - tails[1] >= 2.0,
        "dice": a["metrics"]["dice"] >= b["metrics"]["dice"] - 0.002,
    }
    for k in ("matched_fraction_reference", "matched_fraction_prediction"):
        criteria[k] = a["objects"][k] >= b["objects"][k]
    return dict(
        shared_ids=ids, shared_p95=tails, criteria=criteria, advance=all(criteria.values())
    )


def run(cache, pred, out):
    if not __debug__:
        raise ValueError("Disable optimized Python")
    locked = json.loads(SEAL.read_text())
    ident = identity()
    if (
        locked != json.loads((pred / "predictions.json").read_text())
        or locked["pipeline_id"] != token(ident)
        or locked["identity"] != ident
        or set(locked["specimens"]) != set("235")
        or locked["origin_xyz"] != [360, 376, 256]
        or locked["shape_zyx"] != list(SHAPE)
    ):
        raise ValueError("Different protocol or predictions")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use empty result folder")
    for k in ARMS:
        if sha(pred / f"{k}.npz") != locked["weights"][k]:
            raise ValueError("Different weights")
    # Validate all six sealed masks before reading any transfer label.
    for i in DEV:
        for k in ARMS:
            path = pred / f"sample{i}-{k}.npy"
            a = np.load(path, mmap_mode="r")
            rec = locked["specimens"][str(i)]["masks"][k]
            if (
                a.shape != SHAPE
                or a.dtype != np.bool_
                or sha(path) != rec["file"]
                or digest(a) != rec["mask"]
            ):
                raise ValueError("Mask changed")
    for i in DEV:
        s = locked["specimens"][str(i)]
        _, rs = verified(cache, i, "raw", False)
        seg, ss = verified(cache, i, "segmented", False)
        if (rs["sha256"], ss["sha256"]) != (s["raw"], s["segmented"]):
            raise ValueError("Different sources")
        truth = stack(seg, i, "segmented", *Z)
        tx = label(truth)
        both = {
            k: measure(np.load(pred / f"sample{i}-{k}.npy"), truth, tx, i, out, k) for k in ARMS
        }
        r = dict(
            specimen=i,
            role="development screen",
            pipeline_id=locked["pipeline_id"],
            origin_xyz=locked["origin_xyz"],
            shape_zyx=locked["shape_zyx"],
            reference_mask=digest(truth),
            source=dict(raw=rs["sha256"], segmented=ss["sha256"]),
            arms=both,
            screen=compare(both["control"], both["area"]),
        )
        (out / f"sample{i}.json").write_text(
            json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        print(i, r["screen"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "pred", "out"):
        ap.add_argument("--" + k, required=True, type=Path)
    a = ap.parse_args()
    run(a.cache, a.pred, a.out)
