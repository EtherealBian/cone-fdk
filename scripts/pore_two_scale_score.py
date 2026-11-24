"""Compare the publicly sealed dual-resolution predictions with both comparators."""

import argparse
import json
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_section_score import compare
from pore_transfer import label, measure
from pore_two_scale import E, identity


def run(cache, pred, out):
    if not __debug__:
        raise RuntimeError("Assertions required")
    d = json.loads((E / "predictions.json").read_bytes())
    ident = identity()
    arms = ("census", "fine", "coarse")
    if (
        d != json.loads((pred / "predictions.json").read_bytes())
        or d["identity"] != ident
        or token(ident) != d["pipeline_id"]
        or set(d["specimens"]) != set("235")
        or d["origin_xyz"] != [360, 376, Z[0]]
        or d["shape_zyx"] != list(SHAPE)
    ):
        raise ValueError("Different seal")
    for k, r in d["fits"].items():
        if sha(pred / f"{k}.npz") != r["weights"]:
            raise ValueError("Changed weights")
    for i in DEV:
        for k in arms:
            read_mask(pred, i, k, d["specimens"][str(i)]["masks"][k])
    out.mkdir(parents=True, exist_ok=False)
    for i in DEV:
        r = d["specimens"][str(i)]
        seg, ss = verified(cache, i, "segmented", False)
        _, rs = verified(cache, i, "raw", False)
        if [ss["sha256"], rs["sha256"]] != [r["segmented"], r["raw"]]:
            raise ValueError("Changed source")
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        results = {
            k: measure(read_mask(pred, i, k, r["masks"][k]), t, tx, i, out, k) for k in arms
        }
        screens = {}
        for k in arms[:2]:
            q = compare(results[k], results["coarse"])
            q["criteria"]["decision_replay"] = all(
                a["cpu_gpu_decision_disagreements"] == 0 for a in r["masks"].values()
            )
            q["advance"] = all(q["criteria"].values())
            screens[k] = q
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                role="paired development transfer screen",
                origin_xyz=d["origin_xyz"],
                shape_zyx=d["shape_zyx"],
                pipeline_id=d["pipeline_id"],
                reference_mask=digest(t),
                arms=results,
                screens=screens,
                advance=all(v["advance"] for v in screens.values()),
            ),
        )
        print(i, screens, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "pred", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.pred, a.out)
