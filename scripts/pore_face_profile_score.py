"""Independent source scoring of registered oriented raw-CT face localization."""

import argparse
from fractions import Fraction
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_face_profile import BASE, E, identity
from pore_monotone_score import comparison
from pore_object_catalog import p95
from pore_region import EV, read
from pore_transfer import label, measure


def diagnostic(det, ledger, prev):
    rows = {}
    errs = {k: [] for k in ("detector", "target", "actual")}
    outside = []
    for rid, r in det["references"].items():
        if r["pred"] is None:
            continue
        j = str(r["pred"])
        g = ledger["objects"][j]
        n = r["reference_voxels"]
        old = prev["bound"]["rows"][str(rid)]
        assert (
            g["detector"] == r["voxels"]
            and old["detector_label"] == r["pred"]
            and old["reference"] == n
        )
        lo, hi = old["interval"]
        v = g["target"]
        better = None
        if not lo <= v <= hi:
            outside.append(int(rid))
            better = abs(v - n) < abs(g["detector"] - n)
        for k in errs:
            errs[k].append(Fraction(100 * abs(g[k] - n), n))
        rows[rid] = dict(
            detector_label=r["pred"],
            reference=n,
            target=v,
            actual=g["actual"],
            previous_interval=[lo, hi],
            outside_closer=better,
        )
    return dict(
        rows=rows,
        outside=sorted(outside),
        outside_closer=sum(r["outside_closer"] is True for r in rows.values()),
        p95={k: float(p95(v)) for k, v in errs.items()},
    )


def run(cache, out):
    d = read(E / "predictions.json")
    replay = read(E / "prediction-replay.json")
    ident = identity()
    base = read(BASE / "predictions.json")
    p = {}
    expected = {
        "fit.json",
        "predictions.json",
        *(f"sample{i}-{n}" for i in DEV for n in ("choices.json", "face.npz")),
    }
    assert (
        set(replay) == expected
        and all(sha(E / k) == h for k, h in replay.items())
        and sha(E / "fit.json") == d["fit"]
        and d["identity"] == ident
        and d["pipeline_id"] == token(ident)
        and d["shape_zyx"] == list(SHAPE)
        and d["origin_xyz"] == [360, 376, 256]
        and set(d["specimens"]) == set("235")
    )
    for i in DEV:
        r = d["specimens"][str(i)]
        pr = base["specimens"][str(i)]
        assert all(r[k] == pr[k] for k in ("raw", "segmented"))
        f = E / f"sample{i}-choices.json"
        assert sha(f) == r["ledger"]
        m = read_mask(BASE, i, "curvature", pr["masks"]["curvature"])
        assert digest(m) == r["detector"]
        p[i] = ({"curvature": m, "face": read_mask(E, i, "face", r["masks"]["face"])}, read(f))
    out.mkdir(parents=True, exist_ok=False)
    passed = {}
    for i in DEV:
        fp, ds = verified(cache, i, "segmented", False)
        assert ds["sha256"] == d["specimens"][str(i)]["segmented"]
        t = stack(fp, i, "segmented", *Z)
        tx = label(t)
        ms, g = p[i]
        arms = {k: measure(m, t, tx, i, out, k) for k, m in ms.items()}
        a, b = arms.values()
        c = comparison(a, b)
        c["criteria"].update(
            no_lost=not c["lost"],
            all_p95=b["objects"]["p95_abs_volume_error_pct"]
            <= a["objects"]["p95_abs_volume_error_pct"],
            exact_replay=True,
        )
        c["advance"] = all(c["criteria"].values())
        passed[str(i)] = c["advance"]
        old = EV / "shell-volume" / f"sample{i}.json"
        prev = read(old)
        assert (
            prev["reference_digest"] == digest(t)
            and prev["arms"]["curvature"]["mask_u8_sha256"] == a["mask_u8_sha256"]
        )
        audit = diagnostic(a, g, prev)
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=d["pipeline_id"],
                reference_digest=digest(t),
                arms=arms,
                comparison=c,
                volume_audit=audit,
                previous_score_sha256=sha(old),
                advance=c["advance"],
            ),
        )
        print(
            i,
            passed[str(i)],
            audit["p95"],
            len(audit["outside"]),
            audit["outside_closer"],
            flush=True,
        )
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
