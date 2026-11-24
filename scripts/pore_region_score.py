"""Open transfer references only against a published regional medoid seal."""

import argparse
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_monotone_score import comparison
from pore_region import ARMS, E, identity
from pore_region import read as load
from pore_transfer import label, measure


def run(cache, out):
    seal = load(E / "predictions.json")
    ident = identity()
    prior = load(EVID / "predictions.json")
    assert (
        seal["identity"] == ident
        and seal["pipeline_id"] == token(ident)
        and seal["shape_zyx"] == list(SHAPE)
        and seal["origin_xyz"] == [360, 376, Z[0]]
        and set(seal["specimens"]) == set("235")
    )
    masks = {}
    for i in DEV:
        r = seal["specimens"][str(i)]
        assert (
            sha(E / f"sample{i}-regions.json") == r["ledger_sha256"]
            and sha(E / f"sample{i}-union-pores.csv") == r["union_table"]
        )
        masks[i] = {k: read_mask(E, i, k, r["masks"][k]) for k in ARMS}
        masks[i]["census"] = read_mask(
            EVID, i, "census", prior["specimens"][str(i)]["masks"]["census"]
        )
    out.mkdir(parents=True, exist_ok=False)
    passed = {}
    for i, ms in masks.items():
        r = seal["specimens"][str(i)]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        arms = {k: measure(m, t, tx, i, out, k) for k, m in ms.items()}
        comp = {}
        for k in ("census", "majority"):
            c = comparison(arms[k], arms["medoid"])
            a, b = [arms[name]["objects"]["p95_abs_volume_error_pct"] for name in (k, "medoid")]
            c["criteria"]["all_p95"] = a is not None and b is not None and b <= a
            c["criteria"]["replay"] = all(
                v["cpu_gpu_differences"] == 0 for v in r["masks"].values()
            )
            c["advance"] = all(c["criteria"].values())
            comp[k] = c
        passed[str(i)] = all(c["advance"] for c in comp.values())
        print(i, passed[str(i)], {k: v["shared_p95"] for k, v in comp.items()}, flush=True)
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=seal["pipeline_id"],
                reference_digest=digest(t),
                arms=arms,
                comparisons=comp,
                advance=passed[str(i)],
            ),
        )
    write(
        out / "summary.json",
        dict(pipeline_id=seal["pipeline_id"], advance=all(passed.values()), specimens=passed),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
