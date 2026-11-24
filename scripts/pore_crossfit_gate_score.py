"""Score the three frozen masks on the familiar transfer slab only; no fitting."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_chord_fraction import sources
from pore_crossfit_gate import CAL_SHAPE, PREV, E, donors, identity, names
from pore_monotone_score import comparison
from pore_region import read
from pore_section_gate_score import decisions
from pore_transfer import label, measure


def flips(a, b):
    counts = Counter()
    for j in sorted(a.keys() & b.keys()):
        x, y = a[j], b[j]
        if any(x[k] != y[k] for k in ("detector", "full", "reference")):
            raise ValueError("Mismatched endpoints for common reference")
        r = x["reference"]
        ok = []
        for d in (x, y):
            v = d["full"] if d["selected"] == "face" else d["detector"]
            ok.append(100 * abs(v - r) <= 15 * r)
        counts[f"{x['selected']}_to_{y['selected']}"] += 1
        counts["into_tolerance"] += int(ok == [False, True])
        counts["out_of_tolerance"] += int(ok == [True, False])
        counts["outside_before"] += int(not ok[0])
        counts["outside_after"] += int(not ok[1])
    return dict(counts)


def checked():
    d = read(E / "predictions.json")
    seal = read(E / "prediction-replay.json")
    assert set(seal) == names() and all(sha(E / k) == v for k, v in seal.items())
    ident = identity()
    assert d["identity"] == ident and d["pipeline_id"] == token(ident)
    assert (
        d["shape_zyx"] == list(SHAPE)
        and d["origin_xyz"] == [360, 376, Z[0]]
        and set(d["specimens"]) == set("235")
    )
    assert sha(E / "fit.json") == d["fit"]
    cal = read(E / "fit.json")["calibration"]
    old = read(PREV / "predictions.json")
    masks = {}
    for i in DEV:
        k = str(i)
        fold = cal[k]
        assert (
            fold["donors"] == list(donors(i))
            and fold["held_out"] == i
            and sha(E / f"fold{i}.npz") == fold["weights"]
        )
        p = E / f"sample{i}-oof.npz"
        assert sha(p) == fold["file"]
        with np.load(p, allow_pickle=False) as z:
            m = z["mask"]
            assert m.dtype == bool and m.shape == CAL_SHAPE and digest(m) == fold["mask"]
        (base, end), _, src = sources(i)
        arms = {"curvature": base}
        ledgers = {}
        for doc, folder, arm, tag in [
            (old, PREV, "previous", "gate"),
            (d, E, "crossfit", "crossfit"),
        ]:
            s = doc["specimens"][k]
            assert (s["raw"], s["segmented"]) == tuple(src) and s["parents"] == [
                digest(base),
                digest(end),
            ]
            p = folder / f"sample{i}-choices.json"
            assert sha(p) == s["ledger"]
            ledgers[arm] = read(p)
            arms[arm] = read_mask(folder, i, tag, s["masks"][tag])
        masks[i] = (arms, ledgers)
    return d, cal, masks


def run(cache, out):
    d, cal, masks = checked()
    out.mkdir(parents=True, exist_ok=False)
    passed = {}
    for i in DEV:
        k = str(i)
        path, src = verified(cache, i, "segmented", False)
        assert src["sha256"] == d["specimens"][k]["segmented"]
        truth = stack(path, i, "segmented", *Z)
        tx = label(truth)
        arr, ledgers = masks[i]
        arms = {name: measure(m, truth, tx, i, out, name) for name, m in arr.items()}
        base, old, new = arms.values()
        c = comparison(base, new)
        c["criteria"].update(
            no_lost=not c["lost"],
            all_p95=new["objects"]["p95_abs_volume_error_pct"]
            <= base["objects"]["p95_abs_volume_error_pct"],
            calibration_coverage=all(v["parents"] > 0 for v in cal.values()),
            exact_replay=True,
        )
        c["advance"] = all(c["criteria"].values())
        audits = {name: decisions(base, arms[name], g) for name, g in ledgers.items()}
        rr = audits["crossfit"]["rows"]
        changes = dict(
            previous=flips(audits["previous"]["rows"], rr),
            curvature=flips({j: dict(v, selected="curvature") for j, v in rr.items()}, rr),
        )
        passed[k] = c["advance"]
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=d["pipeline_id"],
                reference_digest=digest(truth),
                arms=arms,
                comparison=c,
                matched_control=comparison(old, new),
                decisions=audits,
                changes=changes,
                advance=c["advance"],
            ),
        )
        print(i, c["shared_p95"], changes, flush=True)
    write(
        out / "summary.json",
        dict(pipeline_id=d["pipeline_id"], specimens=passed, advance=all(passed.values())),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
