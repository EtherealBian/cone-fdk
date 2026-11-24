"""Development-only E/F/G vote diagnostic; reference repair is never a predictor."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_field_replay import masks
from pore_spatial_predict import DEV, ROOT, SHAPE, Z, sha, stack, verified
from pore_transfer import digest, label, measure


def compare(e, f, g, truth):
    if any(a.shape != truth.shape or a.dtype != bool for a in (e, f, g, truth)):
        raise ValueError("Different populations or nonboolean masks")
    code = 4 * e.astype("u1") + 2 * f.astype("u1") + g
    hist = np.bincount(2 * code.ravel() + truth.ravel(), minlength=16).reshape(8, 2)
    disagreement = (code != 0) & (code != 7)
    majority = (e & f) | (e & g) | (f & g)
    repaired = (code == 7) | (disagreement & truth)
    if np.count_nonzero(repaired != truth) != hist[7, 0] + hist[0, 1]:
        raise ValueError("Inconsistent locked error count")
    return hist, majority, repaired


def run(root, cache, out):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Expected empty output")
    for i in DEV:
        src, ss = verified(cache, i, "segmented", False)
        truth = stack(src, i, "segmented", *Z)
        inputs = []
        refs = {}
        for stage in "EFG":
            stem = f"sample{i}"
            folder = root / f"candidate-{stage}"
            r = json.loads((folder / (stem + ".json")).read_text())
            pub = json.loads(
                (
                    ROOT / "pores/evidence" / f"candidate-{stage.lower()}" / (stem + ".json")
                ).read_text()
            )
            path = folder / (stem + "-review.html")
            if (
                r != pub
                or r["role"] != "development"
                or r["specimen"] != i
                or sha(path) != r["review_sha256"]
                or r["source"]["segmented"] != ss
            ):
                raise ValueError("Unexpected development report or viewer")
            from audit import replay

            if replay(folder, i, cache) != (r["method"], r["final_gates"]):
                raise ValueError("Source audit failed")
            layers, shape, origin, num, decode = masks(path)
            if (
                tuple(shape) != SHAPE
                or origin != [360, 376, 400]
                or num != i
                or len(layers) != SHAPE[0]
            ):
                raise ValueError("Different coordinates")
            mask = np.stack(
                [decode(layer["objects"], 65536).reshape(256, 256) != 0 for layer in layers]
            )
            inputs.append(mask)
            refs[stage] = dict(
                pipeline_id=r["pipeline_id"], viewer_sha256=sha(path), mask_sha256=digest(mask)
            )
        hist, majority, repair = compare(*inputs, truth)
        tx = label(truth)
        rec = dict(
            specimen=i,
            role="development diagnostic only",
            source=r["source"],
            origin_xyz=origin,
            shape_zyx=shape,
            inputs=refs,
            mask_reference_sha256=digest(truth),
            vote_order="EFG binary; columns reference background, foreground",
            votes=hist.tolist(),
            fixed_fp=int(hist[7, 0]),
            fixed_fn=int(hist[0, 1]),
            disputed=int(hist[1:7].sum()),
        )
        for k, mask in [("majority", majority), ("reference_assisted", repair)]:
            report = measure(mask, truth, tx, i, out, k)
            gates = report.pop("gates_on_research_slab")
            report["diagnostic_gates"] = gates
            report["matches_over_15pct"] = [
                kk
                for kk, v in report["references"].items()
                if v["voxels"] is not None
                and 100 * abs(v["voxels"] - v["reference_voxels"]) > 15 * v["reference_voxels"]
            ]
            rec[k] = report
            print(
                i,
                k,
                "gates",
                sum(gates.values()),
                "dice",
                report["metrics"]["dice"],
                "P95",
                report["objects"]["p95_abs_volume_error_pct"],
                "over15",
                len(report["matches_over_15pct"]),
                "fixed",
                rec["fixed_fp"],
                rec["fixed_fn"],
                flush=True,
            )
        (out / (stem + ".json")).write_text(
            json.dumps(rec, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    run(a.root, a.cache, a.out)
