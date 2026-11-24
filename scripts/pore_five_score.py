"""Score only the publicly frozen development pilot, never confirmation specimens."""

import argparse
import json
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_five_plane import ARMS, EVID, E, ident
from pore_section_score import compare
from pore_transfer import label, measure


def run(cache, out):
    if not __debug__:
        raise RuntimeError("Assertions required")
    seal = json.loads((E / "predictions.json").read_bytes())
    parent = json.loads((EVID / "predictions.json").read_bytes())
    assert (
        seal["identity"] == ident()
        and seal["pipeline_id"] == token(ident())
        and seal["origin_xyz"] == [360, 376, Z[0]]
        and seal["shape_zyx"] == list(SHAPE)
        and set(seal["specimens"]) == set("235")
    )
    for k in ARMS:
        assert sha(E / f"{k}.npz") == seal["fits"][k]["weights"]
    out.mkdir(parents=True, exist_ok=False)
    for i in DEV:
        r = seal["specimens"][str(i)]
        baseline = parent["specimens"][str(i)]["masks"]["census"]
        m = {
            "census": read_mask(EVID, i, "census", baseline),
            **{k: read_mask(E, i, k, r["masks"][k]) for k in ARMS},
        }
        assert digest(m["census"]) == r["census_mask"]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        arms = {k: measure(mask, t, tx, i, out, k) for k, mask in m.items()}
        screens = {}
        for k in ("census", "slice"):
            q = compare(arms[k], arms["volume"])
            q["criteria"]["decision_replay"] = all(
                v["cpu_gpu_decision_disagreements"] == 0 for v in (baseline, *r["masks"].values())
            )
            q["advance"] = all(q["criteria"].values())
            screens[k] = q
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                role="paired development transfer screen",
                origin_xyz=seal["origin_xyz"],
                shape_zyx=list(SHAPE),
                pipeline_id=seal["pipeline_id"],
                reference_mask=digest(t),
                arms=arms,
                screens=screens,
                advance=all(q["advance"] for q in screens.values()),
            ),
        )
        print(i, screens, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run(a.cache, a.out)
