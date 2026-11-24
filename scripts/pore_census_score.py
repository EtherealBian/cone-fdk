"""Measure only the published matched census/replacement predictions."""

import argparse
import json
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import ARMS, EVID, identity
from pore_section_score import compare
from pore_transfer import label, measure


def run(cache, pred, out):
    if not __debug__:
        raise RuntimeError("Do not disable numerical assertions")
    d = json.loads((EVID / "predictions.json").read_text())
    ident = identity()
    if (
        d != json.loads((pred / "predictions.json").read_text())
        or d["identity"] != ident
        or d["pipeline_id"] != token(ident)
        or d["origin_xyz"] != [360, 376, Z[0]]
        or d["shape_zyx"] != list(SHAPE)
        or set(d["specimens"]) != set("235")
    ):
        raise ValueError("Different seal")
    for k in ARMS:
        if sha(pred / f"{k}.npz") != d["fits"][k]["weights"]:
            raise ValueError("Changed weights")
        for i in DEV:
            read_mask(pred, i, k, d["specimens"][str(i)]["masks"][k])
    out.mkdir(parents=True, exist_ok=False)
    for i in DEV:
        r = d["specimens"][str(i)]
        seg, ss = verified(cache, i, "segmented", False)
        _, rs = verified(cache, i, "raw", False)
        if [r["raw"], r["segmented"]] != [rs["sha256"], ss["sha256"]]:
            raise ValueError("Different sources")
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        arms = {k: measure(read_mask(pred, i, k, r["masks"][k]), t, tx, i, out, k) for k in ARMS}
        cmp = compare(arms[ARMS[0]], arms[ARMS[1]])
        cmp["criteria"]["decision_replay"] = all(
            r["masks"][k]["cpu_gpu_decision_disagreements"] == 0 for k in ARMS
        )
        cmp["advance"] = all(cmp["criteria"].values())
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                role="paired development transfer screen",
                pipeline_id=d["pipeline_id"],
                reference_mask=digest(t),
                origin_xyz=d["origin_xyz"],
                shape_zyx=d["shape_zyx"],
                arms=arms,
                screen=cmp,
            ),
        )
        print(i, cmp, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "pred", "out"):
        p.add_argument("--" + k, required=True, type=Path)
    a = p.parse_args()
    run(a.cache, a.pred, a.out)
