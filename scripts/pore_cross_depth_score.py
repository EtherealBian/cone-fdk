"""Score sealed cross-depth masks against source labels; no label IDs joined across depths."""

import argparse
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_cross_depth import ARMS, MODEL, XYZ, E, Z, identity
from pore_monotone_score import comparison
from pore_region import read
from pore_transfer import label, measure


def run(cache, out):
    seal = read(E / "predictions.json")
    ident = identity()
    src = read(MODEL / "predictions.json")
    assert (
        seal["identity"] == ident
        and seal["pipeline_id"] == token(ident)
        and seal["origin_xyz"] == list(XYZ)
        and seal["shape_zyx"] == list(SHAPE)
        and set(seal["specimens"]) == set("235")
    )
    for k in ARMS:
        assert sha(MODEL / f"{k}.npz") == seal["weights"][k] == src["fits"][k]["weights"]
    ms = {}
    for i in DEV:
        r = seal["specimens"][str(i)]
        assert r["raw"] == src["specimens"][str(i)]["raw"] and all(
            v["cpu_gpu_differences"] == 0 for v in r["masks"].values()
        )
        ms[i] = {k: read_mask(E, i, k, r["masks"][k]) for k in ARMS}
    out.mkdir(parents=True, exist_ok=False)
    cases = {}
    for i in DEV:
        r = seal["specimens"][str(i)]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"] == src["specimens"][str(i)]["segmented"]
        truth = stack(seg, i, "segmented", *Z)
        tx = label(truth)
        arms = {k: measure(ms[i][k], truth, tx, i, out, k) for k in ARMS}
        comp = comparison(arms["center"], arms["profile"])
        diff = {}
        prev = read(MODEL / f"sample{i}.json")
        assert prev["pipeline_id"] == src["pipeline_id"]
        for k in ARMS:
            a, b = prev["arms"][k]["objects"], arms[k]["objects"]
            diff[k] = {}
            for q in ("matched_fraction_reference", "p95_abs_volume_error_pct"):
                diff[k][q] = dict(
                    familiar=a[q],
                    new=b[q],
                    delta=b[q] - a[q] if None not in (a[q], b[q]) else None,
                )
        cases[str(i)] = all(arms["profile"]["gates_on_research_slab"].values())
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                origin_xyz=list(XYZ),
                pipeline_id=seal["pipeline_id"],
                reference_digest=digest(truth),
                arms=arms,
                center_vs_profile=comp,
                depth_comparison=diff,
                replicated=cases[str(i)],
            ),
        )
        print(i, cases[str(i)], arms["profile"]["objects"], flush=True)
    write(
        out / "summary.json",
        dict(pipeline_id=seal["pipeline_id"], replicated=all(cases.values()), specimens=cases),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
