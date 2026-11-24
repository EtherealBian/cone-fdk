"""Score the calibrated fraction and separately compute the explicitly label-aware feasibility bound."""

import argparse
from pathlib import Path

from pore_axial_raw import DEV, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_chord_fraction import FACE_E, E, identity, sources
from pore_fraction_bound import audit
from pore_monotone_score import comparison
from pore_region import read
from pore_transfer import label, measure


def run(cache, out):
    r = read(E / "prediction-replay.json")
    expected = {
        "predictions.json",
        "fit.json",
        *(f"sample{i}-{n}" for i in DEV for n in ("choices.json", "fraction.npz")),
    }
    assert set(r) == expected and all(sha(E / k) == v for k, v in r.items())
    d = read(E / "predictions.json")
    ident = identity()
    assert (
        d["identity"] == ident
        and d["pipeline_id"] == token(ident)
        and d["shape_zyx"] == list(SHAPE)
        and d["origin_xyz"] == [360, 376, Z[0]]
        and set(d["specimens"]) == set("235")
    )
    f = E / "fit.json"
    assert sha(f) == d["fit"]
    cal = read(f)
    allcal = all(s["directional_pairs"] > 0 for s in cal["specimens"].values())
    masks = {}
    for i in DEV:
        (base, full), ledger, source = sources(i)
        p = d["specimens"][str(i)]
        assert (
            source == (p["raw"], p["segmented"])
            and p["parents"] == [digest(base), digest(full)]
            and sha(E / f"sample{i}-choices.json") == p["ledger"]
        )
        masks[i] = (
            {"curvature": base, "fraction": read_mask(E, i, "fraction", p["masks"]["fraction"])},
            ledger,
            digest(full),
        )
    out.mkdir(parents=True, exist_ok=False)
    passed = {}
    population = {}
    for i in DEV:
        path, src = verified(cache, i, "segmented", False)
        assert src["sha256"] == d["specimens"][str(i)]["segmented"]
        t = stack(path, i, "segmented", *Z)
        tx = label(t)
        ms, g, fh = masks[i]
        arms = {k: measure(m, t, tx, i, out, k) for k, m in ms.items()}
        a, b = arms.values()
        c = comparison(a, b)
        c["criteria"].update(
            no_lost=not c["lost"],
            all_p95=b["objects"]["p95_abs_volume_error_pct"]
            <= a["objects"]["p95_abs_volume_error_pct"],
            exact_replay=True,
            calibration_coverage=allcal,
        )
        c["advance"] = all(c["criteria"].values())
        passed[str(i)] = c["advance"]
        old = read(FACE_E / f"sample{i}.json")
        assert (
            old["reference_digest"] == digest(t)
            and not old["comparison"]["gained"]
            and not old["comparison"]["lost"]
            and old["arms"]["face"]["mask_u8_sha256"] == fh
        )
        triples = {}
        for rid, v in a["references"].items():
            h = old["arms"]["face"]["references"][str(rid)]
            assert (v["pred"] is None) == (h["pred"] is None) and v["reference_voxels"] == h[
                "reference_voxels"
            ]
            if v["pred"] is not None:
                actual = g["objects"][str(v["pred"])]["actual"]
                assert actual == h["voxels"]
                triples[str(rid)] = [int(v["voxels"]), int(actual), int(v["reference_voxels"])]
        population[str(i)] = list(triples.values())
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=d["pipeline_id"],
                reference_digest=digest(t),
                arms=arms,
                comparison=c,
                advance=c["advance"],
                fixed_reference_triples=triples,
            ),
        )
        print(i, c["shared_p95"], flush=True)
    bound = audit(population)
    write(
        out / "summary.json",
        dict(
            pipeline_id=d["pipeline_id"],
            specimens=passed,
            advance=all(passed.values()),
            fraction=cal["fraction"],
            optimistic_bound=bound,
        ),
    )
    print(bound, flush=True)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
