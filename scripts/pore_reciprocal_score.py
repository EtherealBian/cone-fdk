"""Compare both whole masks to both source detectors on the fixed transfer population."""

import argparse
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_monotone_score import comparison
from pore_reciprocal import ARMS, E, identity, parents
from pore_region import read
from pore_transfer import label, measure


def run(cache, out):
    d = read(E / "predictions.json")
    ident = identity()
    assert (
        d["identity"] == ident
        and d["pipeline_id"] == token(ident)
        and d["shape_zyx"] == list(SHAPE)
        and d["origin_xyz"] == [360, 376, Z[0]]
        and set(d["specimens"]) == set("235")
    )
    out.mkdir(parents=True, exist_ok=False)
    sm = {}
    allm = {}
    for i in DEV:
        r = d["specimens"][str(i)]
        src, raw, seg = parents(i)
        assert (
            r["raw"] == raw and r["segmented"] == seg and [digest(m) for m in src] == r["parents"]
        )
        allm[i] = dict(zip(("profile", "census"), src, strict=True))
        allm[i].update({k: read_mask(E, i, k, r["masks"][k]) for k in ARMS})
        assert sha(E / f"sample{i}-choices.json") == r["ledger"]
    for i in DEV:
        r = d["specimens"][str(i)]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        arms = {k: measure(m, t, tx, i, out, k) for k, m in allm[i].items()}
        comp = {}
        for k in ARMS:
            comp[k] = {}
            for b in ("profile", "census"):
                c = comparison(arms[b], arms[k])
                p, q = [arms[a]["objects"]["p95_abs_volume_error_pct"] for a in (b, k)]
                c["criteria"].update(
                    no_lost=not c["lost"],
                    all_p95=p is not None and q is not None and q <= p,
                    replay=r["masks"][k]["cpu_gpu_differences"] == 0,
                )
                c["advance"] = all(c["criteria"].values())
                comp[k][b] = c
        sm[str(i)] = comp["forward"]["profile"]["advance"]
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=d["pipeline_id"],
                reference_digest=digest(t),
                arms=arms,
                comparisons=comp,
                advance=sm[str(i)],
            ),
        )
        print(i, sm[str(i)], comp["forward"]["profile"]["shared_p95"], flush=True)
    write(
        out / "summary.json",
        dict(pipeline_id=d["pipeline_id"], specimens=sm, advance=all(sm.values())),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
