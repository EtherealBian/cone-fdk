"""Compare sealed effective-blur predictions with both sharp and unmodified census controls."""

import argparse
import json
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_forward import E
from pore_forward_inverse import ARMS, id_inverse
from pore_section_score import compare
from pore_transfer import label, measure


def run(cache, out):
    seal = json.loads((E / "predictions.json").read_bytes())
    ident = id_inverse()
    prior = json.loads((EVID / "predictions.json").read_bytes())["specimens"]
    assert (
        seal["identity"] == ident
        and seal["pipeline_id"] == token(ident)
        and set(seal["specimens"]) == set("235")
        and seal["origin_xyz"] == [360, 376, 256]
        and seal["shape_zyx"] == list(SHAPE)
    )
    for i in DEV:
        r = seal["specimens"][str(i)]
        for k in ARMS:
            read_mask(E, i, k, r["masks"][k])
            assert r["masks"][k]["cpu_gpu_differences"] == 0
        assert (
            digest(read_mask(EVID, i, "census", prior[str(i)]["masks"]["census"])) == r["census"]
        )
    out.mkdir(parents=True, exist_ok=False)
    summ = {}
    for i in DEV:
        r = seal["specimens"][str(i)]
        sp, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        truth = stack(sp, i, "segmented", *Z)
        tx = label(truth)
        arms = {}
        for k in ("census", *ARMS):
            m = (
                read_mask(EVID, i, k, prior[str(i)]["masks"][k])
                if k == "census"
                else read_mask(E, i, k, r["masks"][k])
            )
            arms[k] = measure(m, truth, tx, i, out, k)
        checks = {}
        for k in ("census", "center"):
            checks[k] = c = compare(arms[k], arms["blur"])
            a, b = [
                {j for j, v in arm["references"].items() if v["pred"] is not None}
                for arm in (arms[k], arms["blur"])
            ]
            c.update(gained=sorted(b - a), lost=sorted(a - b))
            c["criteria"]["replay"] = not any(
                v["cpu_gpu_differences"] for v in r["masks"].values()
            )
            c["advance"] = all(c["criteria"].values())
        summ[str(i)] = {k: v["advance"] for k, v in checks.items()}
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=seal["pipeline_id"],
                reference_digest=digest(truth),
                arms=arms,
                screens=checks,
            ),
        )
        print(i, summ[str(i)], flush=True)
    write(
        out / "summary.json",
        dict(
            pipeline_id=seal["pipeline_id"],
            advance=all(all(v.values()) for v in summ.values()),
            specimens=summ,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Do not run with -O")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    run(a.cache, a.out)
