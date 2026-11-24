"""Read familiar transfer labels only after publication of the complete profile prediction seal."""

import argparse
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_monotone_score import comparison
from pore_region import read
from pore_transfer import label, measure
from pore_volume_curvature import ARMS, BASE, E, identity


def run(cache, out):
    d = read(E / "predictions.json")
    ident = identity()
    assert (
        d["identity"] == ident
        and d["pipeline_id"] == token(ident)
        and list(SHAPE) == d["shape_zyx"]
        and [360, 376, Z[0]] == d["origin_xyz"]
        and set(d["specimens"]) == set("235")
    )
    parent = read(BASE / "predictions.json")
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for k in ARMS:
        assert sha(E / f"{k}.npz") == d["fits"][k]["weights"]
    ms = {
        i: {k: read_mask(E, i, k, d["specimens"][str(i)]["masks"][k]) for k in ARMS} for i in DEV
    }
    for i in DEV:
        r = d["specimens"][str(i)]
        pr = parent["specimens"][str(i)]
        assert r["raw"] == pr["raw"]
        assert r["masks"]["profile"] == pr["masks"]["profile"]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"] == pr["segmented"]
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        arms = {k: measure(v, t, tx, i, out, k) for k, v in ms[i].items()}
        comp = {}
        for k in ("profile",):
            c = comparison(arms[k], arms["curvature"])
            a, b = [arms[j]["objects"]["p95_abs_volume_error_pct"] for j in (k, "curvature")]
            c["criteria"]["no_lost"] = not c["lost"]
            c["criteria"]["all_p95"] = a is not None and b is not None and b <= a
            c["criteria"]["replay"] = all(
                v["cpu_gpu_differences"] == 0 for v in r["masks"].values()
            )
            c["advance"] = all(c["criteria"].values())
            comp[k] = c
        summary[str(i)] = all(c["advance"] for c in comp.values())
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=d["pipeline_id"],
                reference_digest=digest(t),
                arms=arms,
                comparisons=comp,
                advance=summary[str(i)],
            ),
        )
        print(i, summary[str(i)], {k: v["shared_p95"] for k, v in comp.items()}, flush=True)
    write(
        out / "summary.json",
        dict(pipeline_id=d["pipeline_id"], specimens=summary, advance=all(summary.values())),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("cache", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
