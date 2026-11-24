"""CPU-only, read-only replay of split decisions, cuts and all published native tables."""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pores"))
from pore_check import reference
from pore_split import DEV, SEALED, SHAPE, Z, split
from pore_split_score import decision, ledger
from pore_transfer import digest, label, pairs, sha, stack, verified
from validate import SPEC, final_gates, objects, score


def require(ok, message):
    if not ok:
        raise ValueError(message)


def replay_split(folder, cache):
    seal = json.loads(SEALED.read_text())
    records = {}
    for i in DEV:
        s = f"sample{i}"
        r = json.loads((folder / (s + ".json")).read_text())
        q = seal["arms"][str(i)]
        origin = seal["baseline"]["specimens"][str(i)]
        records[str(i)] = r
        with np.load(folder / (s + "-probability.npz"), allow_pickle=False) as arr:
            p = arr["probability"]
        require(
            p.shape == SHAPE
            and p.dtype == np.dtype("<f4")
            and digest(p) == origin["probability"],
            "Unsealed probability field",
        )
        base = p >= 0.5
        arm, rows = split(p)
        require(q["base"] == digest(base) and q["arm"] == digest(arm), "Unsealed masks")
        path, source = verified(cache, i, "segmented", False)
        require(source["sha256"] == origin["segmented"], "Different annotation archive")
        truth = stack(path, i, "segmented", *Z)
        require(
            r["specimen"] == i
            and r["shape_zyx"] == list(SHAPE)
            and r["origin_xyz"] == [360, 376, Z[0]]
            and r["reference"] == digest(truth),
            "Different evaluation region",
        )
        require(r["ledger"] == ledger(base, arm, truth, rows), "Wrong intervention outcomes")
        tx = label(truth)
        for name, mask in [("base", base), ("arm", arm)]:
            a = r[name]
            table = folder / f"{s}-{name}-pores.csv"
            require(
                digest(mask) == a["mask_u8_sha256"]
                and sha(table) == a["table_sha256"]
                and a["slab_depths"] == [8, 37],
                "Wrong native table/mask",
            )
            lab, rr = reference(mask, 6, (SPEC[i][2],) * 3, table, a["native"])
            require(len(rr) == a["exact_components"], "Wrong object count")
            with TemporaryDirectory() as tmp:
                m = Path(tmp) / "matches.csv"
                o = objects(lab, truth, rr, m)
                require(
                    m.read_bytes() == (folder / f"{s}-{name}-matches.csv").read_bytes(),
                    "Different one-to-one matches",
                )
            pred = label(mask)
            matched = pairs(pred, tx, 64)
            refs = {
                str(k): dict(
                    reference_voxels=int(tx[1][k]),
                    pred=matched[k][0] if k in matched else None,
                    voxels=int(pred[1][matched[k][0]]) if k in matched else None,
                    iou=matched[k][1] if k in matched else None,
                )
                for k in np.flatnonzero(tx[2] & (tx[1] >= 64))
            }
            require(refs == a["references"], "Different full reference ledger")
            met = score(mask, truth)
            require(
                o == a["objects"]
                and met == a["metrics"]
                and final_gates(met, o) == a["gates_on_research_slab"],
                "Different unrounded gates",
            )
        print("split reference replay", i, r["ledger"]["counts"], flush=True)
    require(
        decision(records) == json.loads((folder / "decision.json").read_text()),
        "Wrong promotion decision",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    replay_split(a.folder, a.cache)
