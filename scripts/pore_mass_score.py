"""Score all three sealed gray-integration arms on previously observed development data."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_mass import ARMS, SEED, E, identity
from pore_transfer import label, measure


def compare(base, new):
    x, y = [s["references"] for s in (base, new)]
    a, b = [{j for j, v in m.items() if v["pred"] is not None} for m in (x, y)]
    common = sorted(a & b)
    tails = [
        float(
            np.percentile(
                [
                    100
                    * abs(m[j]["voxels"] - m[j]["reference_voxels"])
                    / m[j]["reference_voxels"]
                    for j in common
                ],
                95,
            )
        )
        if common
        else None
        for m in (x, y)
    ]
    return dict(common=common, paired_p95=tails, gained=sorted(b - a), lost=sorted(a - b))


def score(cache, out):
    seal = json.loads((E / "predictions.json").read_bytes())
    ident = identity()
    assert (
        seal["pipeline_id"] == token(ident)
        and seal["identity"] == ident
        and set(seal["specimens"]) == set("235")
    )
    assert seal["origin_xyz"] == [360, 376, 256] and seal["shape_zyx"] == list(SHAPE)
    prior = json.loads((EVID / "predictions.json").read_bytes())["specimens"]
    for i in DEV:
        rec = seal["specimens"][str(i)]
        assert sha(E / f"sample{i}-estimates.json") == rec["estimates"]
        for k in ARMS:
            read_mask(E, i, k, rec["masks"][k])
        assert (
            digest(read_mask(EVID, i, "census", prior[str(i)]["masks"]["census"]))
            == rec["census"]
        )
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads((SEED / "overlap.json").read_bytes())["specimens"]
    data = {}
    for i in DEV:
        rec = seal["specimens"][str(i)]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == rec["segmented"]
        truth = stack(seg, i, "segmented", *Z)
        tx = label(truth)
        assert digest(truth) == old[str(i)]["reference_mask"]
        rows = json.loads((E / f"sample{i}-estimates.json").read_bytes())
        prev = old[str(i)]["arms"]["gray"]["rows"]
        byid = {r["seed"]: r for r in rows if r["original"] < 64 <= r["prior_gray"]}
        assert set(byid) == {r["seed"] for r in prev}
        arms = {}
        cmp = {}
        summaries = {}
        for k in ("census", *ARMS):
            mask = (
                read_mask(EVID, i, k, prior[str(i)]["masks"][k])
                if k == "census"
                else read_mask(E, i, k, rec["masks"][k])
            )
            arms[k] = ar = measure(mask, truth, tx, i, out, k)
            cmp[k] = c = compare(arms["census"], ar)
            obj = ar["objects"]
            cen = arms["census"]["objects"]
            summaries[k] = dict(
                matches=obj["matches"],
                predictions=obj["eligible_prediction"],
                references=obj["eligible_reference"],
                p95=obj["p95_abs_volume_error_pct"],
                paired_p95=c["paired_p95"],
                gained=c["gained"],
                lost=c["lost"],
                dice=ar["metrics"]["dice"],
                passed=sum(ar["gates_on_research_slab"].values()),
                change_unmatched=obj["eligible_prediction"]
                - obj["matches"]
                - cen["eligible_prediction"]
                + cen["matches"],
            )
            if k in ("size", "mass"):
                active = [r for r in prev if byid[r["seed"]][k]]
                assert not c["lost"] and set(c["gained"]) == {
                    r["matched"] for r in active if r["matched"] is not None
                }
                assert len(active) == obj["eligible_prediction"] - cen["eligible_prediction"]
                summaries[k].update(
                    promotions=len(active),
                    below_rejected=sum(
                        not byid[r["seed"]][k] and r["category"] == "below 64" for r in prev
                    ),
                )
        entrants = [
            dict(
                seed=q["seed"],
                old_reference=q["largest_ref"],
                old_match=q["matched"],
                old_category=q["category"],
                mass=byid[q["seed"]]["mass"],
                size=byid[q["seed"]]["size"],
            )
            for q in prev
        ]
        data[str(i)] = summaries
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=seal["pipeline_id"],
                reference_digest=digest(truth),
                arms=arms,
                comparisons=cmp,
                previous_entrants=entrants,
            ),
        )
        print(i, summaries, flush=True)
    checks = dict(
        rescue_two=sum(bool(d["mass"]["gained"]) for d in data.values()) >= 2,
        no_new_unmatched=all(d["mass"]["change_unmatched"] == 0 for d in data.values()),
        beat_size=sum(len(d["mass"]["gained"]) - len(d["size"]["gained"]) for d in data.values())
        > 0,
    )
    bchecks = dict(
        mass_pass=all(checks.values()),
        no_lost=all(not d["boundary"]["lost"] for d in data.values()),
        coverage=all(
            d["boundary"]["matches"] * d["census"][den]
            >= d["census"]["matches"] * d["boundary"][den]
            for d in data.values()
            for den in ("predictions", "references")
        ),
        paired_nonincreasing=all(
            None not in (t := d["boundary"]["paired_p95"]) and t[1] <= t[0] for d in data.values()
        ),
        paired_improved_two=sum(
            None not in (t := d["boundary"]["paired_p95"]) and t[1] < t[0] for d in data.values()
        )
        >= 2,
    )
    write(
        out / "summary.json",
        dict(
            pipeline_id=seal["pipeline_id"],
            previous_overlap_sha256=sha(SEED / "overlap.json"),
            development=data,
            mass_screen=checks,
            boundary_screen=bchecks,
            advance=all(bchecks.values()),
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Do not run with -O")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    score(a.cache, a.out)
