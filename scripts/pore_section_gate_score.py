"""Independently score the sealed section-based gate, including its actual endpoint decisions."""

import argparse
from collections import Counter
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_chord_fraction import sources
from pore_monotone_score import comparison
from pore_region import read
from pore_section_gate import E, identity
from pore_transfer import label, measure


def decisions(old, new, g):
    choices = {}
    counts = Counter()
    for j, row in old["references"].items():
        if row["pred"] is None or new["references"][j]["pred"] is None:
            continue
        d = g["objects"][str(row["pred"])]
        v, u, r = d["detector"], d["full"], row["reference_voxels"]
        assert v == row["voxels"] and d["actual"] == new["references"][j]["voxels"]
        a, b = abs(v - r), abs(u - r)
        best = "tie" if a == b else "face" if b < a else "curvature"
        status = "tie" if best == "tie" else "correct" if best == d["endpoint"] else "wrong"
        counts[status] += 1
        counts["strict_face_better"] += int(b < a)
        counts["strict_detector_better"] += int(a < b)
        choices[str(j)] = dict(
            detector=v, full=u, reference=r, selected=d["endpoint"], best=best, status=status
        )
    return dict(counts=dict(counts), rows=choices)


def run(cache, out):
    r = read(E / "prediction-replay.json")
    expected = {
        "fit.json",
        "predictions.json",
        *(f"sample{i}-{k}" for i in DEV for k in ("gate.npz", "choices.json")),
    }
    assert set(r) == expected and all(sha(E / k) == v for k, v in r.items())
    d = read(E / "predictions.json")
    ident = identity()
    f = E / "fit.json"
    assert (
        d["identity"] == ident
        and d["pipeline_id"] == token(ident)
        and d["fit"] == sha(f)
        and d["shape_zyx"] == list(SHAPE)
        and d["origin_xyz"] == [360, 376, Z[0]]
        and set(d["specimens"]) == set("235")
    )
    cal = read(f)["calibration"]
    coverage = all(cal[str(i)]["parents"] > 0 for i in DEV)
    masks = {}
    for i in DEV:
        (m, e), _, src = sources(i)
        s = d["specimens"][str(i)]
        p = E / f"sample{i}-choices.json"
        assert (
            tuple(src) == (s["raw"], s["segmented"])
            and [digest(m), digest(e)] == s["parents"]
            and sha(p) == s["ledger"]
        )
        masks[i] = (m, read_mask(E, i, "gate", s["masks"]["gate"]), read(p))
    out.mkdir(parents=True, exist_ok=False)
    passed = {}
    for i in DEV:
        path, s = verified(cache, i, "segmented", False)
        assert s["sha256"] == d["specimens"][str(i)]["segmented"]
        t = stack(path, i, "segmented", *Z)
        tx = label(t)
        m, p, g = masks[i]
        arms = {k: measure(v, t, tx, i, out, k) for k, v in [("curvature", m), ("gate", p)]}
        a, b = arms.values()
        c = comparison(a, b)
        c["criteria"].update(
            no_lost=not c["lost"],
            all_p95=b["objects"]["p95_abs_volume_error_pct"]
            <= a["objects"]["p95_abs_volume_error_pct"],
            calibration_coverage=coverage,
            exact_replay=True,
        )
        c["advance"] = all(c["criteria"].values())
        passed[str(i)] = c["advance"]
        audit = decisions(a, b, g)
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=d["pipeline_id"],
                reference_digest=digest(t),
                arms=arms,
                comparison=c,
                decisions=audit,
                advance=c["advance"],
            ),
        )
        print(i, c["shared_p95"], audit["counts"], flush=True)
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
