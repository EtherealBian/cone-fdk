"""Evaluate published orientation means, including errors even all eight views agree on."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_monotone_score import comparison
from pore_orientation import ARMS, E, identity
from pore_transfer import label, measure


def ceiling(v, truth):
    fg = np.bincount(v[truth], minlength=9).tolist()
    bg = np.bincount(v[~truth], minlength=9).tolist()
    fn = fg[0]
    fp = bg[8]
    tp = sum(fg) - fn
    return dict(
        reference_pore=fg,
        reference_material=bg,
        unanimous_fn=fn,
        unanimous_fp=fp,
        dice_ceiling=2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp else None,
    )


def run(cache, out):
    seal = json.loads((E / "predictions.json").read_bytes())
    ident = identity()
    assert (
        ident == seal["identity"]
        and token(ident) == seal["pipeline_id"]
        and seal["shape_zyx"] == list(SHAPE)
        and seal["origin_xyz"] == [360, 376, 256]
        and set(seal["specimens"]) == set("235")
    )
    for i in DEV:
        r = seal["specimens"][str(i)]
        assert r["identity_parent_differences"] == 0
        for k in ARMS:
            read_mask(E, i, k, r["masks"][k])
        f = E / f"sample{i}-votes.npz"
        assert sha(f) == r["votes_file"]
        with np.load(f, allow_pickle=False) as f:
            v = f["votes"]
            assert (
                v.shape == SHAPE
                and v.dtype == np.uint8
                and digest(v) == r["votes_u8"]
                and np.bincount(v.ravel(), minlength=9).tolist() == r["vote_counts"]
            )
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for i in DEV:
        r = seal["specimens"][str(i)]
        sp, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        t = stack(sp, i, "segmented", *Z)
        tx = label(t)
        m = {k: read_mask(E, i, k, r["masks"][k]) for k in ARMS}
        arms = {k: measure(v, t, tx, i, out, k) for k, v in m.items()}
        base, mean = m.values()
        change = base != mean
        with np.load(E / f"sample{i}-votes.npz", allow_pickle=False) as f:
            v = f["votes"]
            assert not np.any(change & ((v == 0) | (v == 8)))
            votes = ceiling(v, t)
        c = comparison(arms["identity"], arms["mean8"])
        c["criteria"]["replay"] = all(not z["cpu_gpu_differences"] for z in r["masks"].values())
        c["advance"] = all(c["criteria"].values())
        summary[str(i)] = c["advance"]
        edits = {
            k: int(np.sum(change & cond))
            for k, cond in [
                ("new_fp", mean & ~t),
                ("new_fn", ~mean & t),
                ("fixed_fp", ~mean & ~t),
                ("fixed_fn", mean & t),
            ]
        }
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=seal["pipeline_id"],
                reference_digest=digest(t),
                arms=arms,
                screen=c,
                edits=edits,
                votes=votes,
            ),
        )
        print(i, c["advance"], c["shared_p95"], votes["dice_ceiling"], flush=True)
    write(
        out / "summary.json",
        dict(pipeline_id=seal["pipeline_id"], advance=all(summary.values()), specimens=summary),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    run(a.cache, a.out)
