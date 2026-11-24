"""Score only published detector-locked masks, including an explicitly label-aware interval bound."""

import argparse
from fractions import Fraction
from pathlib import Path

from pore_axial_raw import DEV, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_monotone_score import comparison
from pore_object_catalog import p95
from pore_region import read
from pore_shell_volume import E, identity, parents
from pore_transfer import label, measure


def bounds(arm, ledger):
    rows = {}
    e = {k: [] for k in ("detector", "target", "applied", "interval")}
    for rid, r in arm["references"].items():
        if r["pred"] is None:
            continue
        g = ledger["objects"][str(r["pred"])]
        ref = r["reference_voxels"]
        v = g["detector"]
        assert v == r["voxels"]
        pool = ([x["size"] for x in g["donors"]] + [v]) if len(g["donors"]) >= 2 else [v]
        lo, hi = min(pool), max(pool)
        best = min(hi, max(ref, lo))
        es = {}
        for k, val in zip(e, (v, g["target"], g["actual"], best), strict=True):
            a = Fraction(100 * abs(val - ref), ref)
            e[k].append(a)
            es[k] = str(a)
        rows[str(rid)] = dict(
            detector_label=r["pred"],
            reference=ref,
            interval=[lo, hi],
            best=best,
            percent_errors=es,
        )
    return dict(
        rows=rows,
        p95={k: float(p95(v)) if v else None for k, v in e.items()},
        outside_15=sorted(
            int(k) for k, r in rows.items() if Fraction(r["percent_errors"]["interval"]) > 15
        ),
    )


def run(cache, out):
    d = read(E / "predictions.json")
    ident = identity()
    replay = read(E / "prediction-replay.json")
    masks = {}
    ledgers = {}
    expected = {
        "predictions.json",
        *(f"sample{i}-{t}" for i in DEV for t in ("choices.json", "shell.npz")),
    }
    assert set(replay) == expected and all(sha(E / k) == v for k, v in replay.items())
    assert (
        d["identity"] == ident
        and d["pipeline_id"] == token(ident)
        and set(d["specimens"]) == set("235")
        and d["origin_xyz"] == [360, 376, 256]
        and d["shape_zyx"] == [96, 256, 256]
    )
    for i in DEV:
        ms, raw, seg = parents(i)
        r = d["specimens"][str(i)]
        assert (r["raw"], r["segmented"]) == (raw, seg) and r["parents"] == [
            digest(m) for m in ms
        ]
        f = E / f"sample{i}-choices.json"
        assert sha(f) == r["ledger"]
        ledgers[i] = read(f)
        masks[i] = {"curvature": ms[0], "shell": read_mask(E, i, "shell", r["masks"]["shell"])}
    out.mkdir(parents=True, exist_ok=False)
    passed = {}
    for i in DEV:
        path, s = verified(cache, i, "segmented", False)
        assert s["sha256"] == d["specimens"][str(i)]["segmented"]
        t = stack(path, i, "segmented", *Z)
        tx = label(t)
        arms = {k: measure(m, t, tx, i, out, k) for k, m in masks[i].items()}
        a, b = arms.values()
        c = comparison(a, b)
        c["criteria"].update(
            no_lost=not c["lost"],
            all_p95=b["objects"]["p95_abs_volume_error_pct"]
            <= a["objects"]["p95_abs_volume_error_pct"],
            exact_prediction_replay=True,
        )
        c["advance"] = all(c["criteria"].values())
        passed[str(i)] = c["advance"]
        bound = bounds(a, ledgers[i])
        assert abs(bound["p95"]["detector"] - a["objects"]["p95_abs_volume_error_pct"]) < 1e-8
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=d["pipeline_id"],
                reference_digest=digest(t),
                arms=arms,
                comparison=c,
                bound=bound,
                advance=c["advance"],
            ),
        )
        print(i, c["advance"], c["gained"], c["lost"], bound["p95"], flush=True)
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
