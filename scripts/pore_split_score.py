"""Score the prospectively sealed two-core pilot. No holdout path is accepted."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_bridge_review import compare
from pore_split import DEV, SEALED, SHAPE, Z, load_prob, split
from pore_transfer import digest, label, measure, stack, verified


def ledger(base, arm, truth, rows):
    pl, al = label(base)[0], label(arm)[0]
    from pore_transfer import pairs

    tx = label(truth)
    bm = pairs(label(base), tx, 64)
    am = pairs(label(arm), tx, 64)
    bmap = {p: r for r, (p, _) in bm.items()}
    amap = {p: r for r, (p, _) in am.items()}
    audit = compare(pl, rows, truth)
    counts = dict(
        events=0,
        flagged=0,
        edited=0,
        rescued=0,
        unsupported=0,
        single_split=0,
        unresolved_events=0,
    )
    for row in audit["rows"]:
        p = row["pred"]
        ever = row["event"]
        edit = row["decision"] == "edited"
        counts["events"] += ever
        counts["edited"] += edit
        counts["flagged"] += row["flagged"]
        daughters = [int(al.flat[t]) for t in row["daughter_seeds"]]
        linked = [amap.get(t) for t in daughters]
        covers = {v["reference"] for v in row["covered_references"]}
        rescued = edit and ever and len(set(linked) & covers) >= 2
        unsupported = edit and not ever
        false_split = unsupported and p in bmap and len(set(linked) - {None}) < 2
        row.update(
            baseline_match=bmap.get(p),
            daughter_matches=linked,
            rescued=rescued,
            unsupported=unsupported,
            single_split=false_split,
        )
        counts["rescued"] += rescued
        counts["unsupported"] += unsupported
        counts["single_split"] += false_split
        counts["unresolved_events"] += ever and not rescued
    return dict(
        counts=counts,
        alerts=audit["counts"],
        parents=audit["rows"],
        lost_reference_ids=sorted(bm.keys() - am.keys()),
        gained_reference_ids=sorted(am.keys() - bm.keys()),
    )


def decision(records):
    checked = {}
    for i in DEV:
        r = records[str(i)]
        stats = r["ledger"]
        b, a = (r[k]["objects"] for k in ("base", "arm"))
        p, q = b["p95_abs_volume_error_pct"], a["p95_abs_volume_error_pct"]
        checked[str(i)] = dict(
            no_unsupported=not stats["counts"]["unsupported"],
            no_lost_ids=not stats["lost_reference_ids"],
            unmatched_not_worse=a["eligible_prediction"] - a["matches"]
            <= b["eligible_prediction"] - b["matches"],
            p95_not_worse=p is not None and q is not None and q <= p,
        )
    support = sum(records[str(i)]["ledger"]["counts"]["rescued"] > 0 for i in DEV) >= 2
    clean = all(all(vals.values()) for vals in checked.values())
    return dict(
        criteria_by_specimen=checked,
        two_rescues_on_two_specimens=support,
        decision="fail" if not clean else "pass" if support else "insufficient support",
    )


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def run(folder, cache, out):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new empty evidence directory")
    sealed = json.loads(SEALED.read_text())
    baseline = sealed["baseline"]
    local = json.loads((folder / "baseline.json").read_text())
    records = {}
    if local != baseline:
        raise ValueError("Different published manifest")
    # Validate all three baseline and edited masks before opening even the first new label.
    for i in DEV:
        p = load_prob(folder, baseline, i)
        arm, rows = split(p)
        locked = sealed["arms"][str(i)]
        if locked != dict(
            base=digest(p >= 0.5),
            arm=digest(arm),
            eligible=len(rows),
            flagged=sum(r["flagged"] for r in rows),
            edited=sum(r["decision"] == "edited" for r in rows),
        ):
            raise ValueError("Mask differs from public seal")
    for i in DEV:
        p = load_prob(folder, baseline, i)
        base = p >= 0.5
        arm, rows = split(p)
        path, source = verified(cache, i, "segmented", False)
        if source["sha256"] != baseline["specimens"][str(i)]["segmented"]:
            raise ValueError("Changed annotation archive")
        truth = stack(path, i, "segmented", *Z)
        tx = label(truth)
        detail = ledger(base, arm, truth, rows)
        record = dict(
            specimen=i,
            origin_xyz=[360, 376, Z[0]],
            shape_zyx=list(SHAPE),
            reference=digest(truth),
            ledger=detail,
        )
        for name, mask in [("base", base), ("arm", arm)]:
            record[name] = measure(mask, truth, tx, i, out, name)
        np.savez_compressed(out / f"sample{i}-probability.npz", probability=p)
        dump(out / f"sample{i}.json", record)
        records[str(i)] = record
        print(i, detail["counts"], record["arm"]["objects"], flush=True)
    result = decision(records)
    dump(out / "decision.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["baseline", "cache", "out"]:
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    run(a.baseline, a.cache, a.out)
