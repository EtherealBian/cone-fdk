"""Compare sealed gray and equal-size geometry arms to census with every entrant retained."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_small_seed import ARMS, E, identity
from pore_transfer import label, measure


def run(cache, out):
    ident = identity()
    seal = json.loads((E / "predictions.json").read_bytes())
    base = json.loads((EVID / "predictions.json").read_bytes())
    assert (
        seal["identity"] == ident
        and seal["pipeline_id"] == token(ident)
        and set(seal["specimens"]) == set("235")
        and seal["origin_xyz"] == [360, 376, 256]
        and seal["shape_zyx"] == list(SHAPE)
    )
    out.mkdir(parents=True, exist_ok=False)
    # Verify all locked proposals before any label is read.
    for i in DEV:
        r = seal["specimens"][str(i)]
        assert (
            sha(E / f"sample{i}-seeds.json") == r["seeds_sha256"]
            and r["growth_replay_differences"] == r["median_cpu_differences"] == 0
        )
        for k in ARMS:
            read_mask(E, i, k, r["masks"][k])
    data = {}
    for i in DEV:
        r = seal["specimens"][str(i)]
        seeds = json.loads((E / f"sample{i}-seeds.json").read_bytes())
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        m0 = read_mask(EVID, i, "census", base["specimens"][str(i)]["masks"]["census"])
        assert digest(m0) == r["census"]
        orig = measure(m0, t, tx, i, out, "census")
        baseline = {j for j, q in orig["references"].items() if q["pred"] is not None}
        arms = {}
        reviews = {}
        for k in ARMS:
            m = read_mask(E, i, k, r["masks"][k])
            arms[k] = a = measure(m, t, tx, i, out, k)
            lab, sizes, whole, boxes = label(m)
            refs = a["references"]
            allids = {j for j, v in refs.items() if v["pred"] is not None}
            common = sorted(baseline & allids)
            lookup = {v["pred"]: j for j, v in refs.items() if v["pred"] is not None}
            entered = []
            for s in seeds:
                if s["final"] < 64:
                    continue
                idx = int(lab[tuple(s["seed_zyx"])])
                assert sizes[idx] == s["final"] and whole[idx]
                j = lookup.get(idx)
                box = boxes[idx - 1]
                xyz = np.argwhere(lab[box] == idx)[0] + [v.start for v in box]
                entered.append(
                    dict(
                        original_seed=s["seed_id"],
                        final_id=int(np.ravel_multi_index(xyz, SHAPE)) + 1,
                        original=s["original"],
                        final=s["final"],
                        matched_reference=j,
                        iou=refs[j]["iou"] if j else None,
                        new_reference=j in allids - baseline,
                    )
                )
            assert (
                len({w["final_id"] for w in entered})
                == len(entered)
                == a["objects"]["eligible_prediction"] - orig["objects"]["eligible_prediction"]
            )
            tails = [
                float(
                    np.percentile(
                        [
                            100
                            * abs(x[j]["reference_voxels"] - x[j]["voxels"])
                            / x[j]["reference_voxels"]
                            for j in common
                        ],
                        95,
                    )
                )
                if common
                else None
                for x in (orig["references"], refs)
            ]
            reviews[k] = dict(
                entrants=entered,
                gained=sorted(allids - baseline),
                lost=sorted(baseline - allids),
                common=common,
                shared_p95=tails,
                extra_unmatched=sum(w["matched_reference"] is None for w in entered),
            )
        assert {s["original_seed"]: s["final"] for s in reviews["gray"]["entrants"]} == {
            s["original_seed"]: s["final"] for s in reviews["geometry"]["entrants"]
        }
        improved = [
            j
            for j in reviews["gray"]["gained"]
            if arms["gray"]["references"][j]["iou"]
            > (arms["geometry"]["references"][j]["iou"] or 0)
        ]
        data[str(i)] = dict(
            gained=reviews["gray"]["gained"],
            lost=reviews["gray"]["lost"],
            extra_unmatched=reviews["gray"]["extra_unmatched"],
            shared_p95=reviews["gray"]["shared_p95"],
            improved_iou=improved,
        )
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=seal["pipeline_id"],
                reference_mask=digest(t),
                origin_xyz=seal["origin_xyz"],
                shape_zyx=list(SHAPE),
                arms=dict(census=orig, **arms),
                comparisons=reviews,
                improved_iou=improved,
            ),
        )
        print(i, data[str(i)], flush=True)
    crit = dict(
        rescue_two_specimens=sum(bool(q["gained"]) for q in data.values()) >= 2,
        no_new_unmatched=not any(q["extra_unmatched"] for q in data.values()),
        no_lost_matches=not any(q["lost"] for q in data.values()),
        common_p95=all(
            None not in q["shared_p95"] and q["shared_p95"][1] <= q["shared_p95"][0]
            for q in data.values()
        ),
        outperform_geometry=any(q["improved_iou"] for q in data.values()),
    )
    write(out / "summary.json", dict(criteria=crit, advance=all(crit.values()), development=data))
    print(crit)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run(a.cache, a.out)
