"""Score both axial-context research arms only after the public development seal."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import (
    ARMS,
    DEV,
    SEAL,
    SHAPE,
    Z,
    digest,
    identity,
    sha,
    stack,
    token,
    verified,
    write,
)
from pore_section_score import compare
from pore_transfer import label, measure


def read_mask(pred, i, k, r):
    p = pred / f"sample{i}-{k}.npz"
    if sha(p) != r["file"]:
        raise ValueError("Changed mask file")
    with np.load(p, allow_pickle=False) as v:
        m = v["mask"]
    if m.shape != SHAPE or m.dtype != np.bool_ or digest(m) != r["mask"]:
        raise ValueError("Changed mask")
    return m


def run(cache, pred, out):
    if not __debug__:
        raise RuntimeError("Do not use -O")
    d = json.loads(SEAL.read_text())
    ident = identity()
    specs = d["specimens"]
    if (
        d != json.loads((pred / "predictions.json").read_text())
        or set(specs) != set("235")
        or d["identity"] != ident
        or d["pipeline_id"] != token(ident)
        or d["origin_xyz"] != [360, 376, 256]
        or d["shape_zyx"] != list(SHAPE)
    ):
        raise ValueError("Unknown seal or protocol")
    for k in ARMS:
        if sha(pred / f"{k}.npz") != d["fits"][k]["weights"]:
            raise ValueError("Changed weights")
        for i in DEV:
            read_mask(pred, i, k, specs[str(i)]["masks"][k])
    out.mkdir(parents=True, exist_ok=False)
    for i in DEV:
        rec = specs[str(i)]
        seg, ss = verified(cache, i, "segmented", False)
        _, rs = verified(cache, i, "raw", False)
        if [ss["sha256"], rs["sha256"]] != [rec["segmented"], rec["raw"]]:
            raise ValueError("Changed source")
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        arms = {
            k: measure(read_mask(pred, i, k, rec["masks"][k]), t, tx, i, out, k) for k in ARMS
        }
        cmp = compare(arms["plane"], arms["axial"])
        cmp["criteria"]["decision_replay"] = all(
            rec["masks"][k]["cpu_gpu_decision_disagreements"] == 0 for k in ARMS
        )
        cmp["advance"] = all(cmp["criteria"].values())
        print(i, cmp, flush=True)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "pred", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    run(a.cache, a.pred, a.out)
