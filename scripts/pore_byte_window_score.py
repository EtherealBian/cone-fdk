"""Measure sealed byte-table predictions and compute a separate exact target-assisted bound."""

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, PTX, SHAPE, Denoise, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_byte_window import ARMS, E, census, decode, graykeys, identity, lookup
from pore_monotone_score import comparison
from pore_region import read
from pore_transfer import label, measure


def bound(n, p):
    n = np.asarray(n, dtype="i8")
    p = np.asarray(p, dtype="i8")
    if not (
        n.ndim == 1
        and len(n)
        and n.shape == p.shape
        and np.all((0 <= p) & (p <= n) & (n > 0))
        and int(n.sum()) <= 2**26
    ):
        raise ValueError("Invalid counts")
    total = int(p.sum())
    if not total:
        rat = Fraction(1)
        best = 0
    else:
        order = sorted(range(len(p)), key=lambda j: Fraction(int(p[j]), int(n[j])), reverse=True)
        num, den = 0, total
        twice_tp, called = 0, 0
        for j in order:
            twice_tp += 2 * int(p[j])
            called += int(n[j])
            if twice_tp * den > num * (total + called):
                num, den = twice_tp, total + called
        rat = Fraction(num, den)
        # Independent optimality certificate for all subsets, not only prefixes.
        surplus = 2 * p * rat.denominator - rat.numerator * n
        best = int((surplus > 0).sum())
        assert int(np.maximum(surplus, 0).sum()) == rat.numerator * total
    return dict(
        keys=len(p),
        conflicting=int(((p > 0) & (p < n)).sum()),
        heterogeneous_centers=int(n[(p > 0) & (p < n)].sum()),
        unavoidable_voxel_errors=int(np.minimum(p, n - p).sum()),
        maximum_dice=str(rat),
        maximum_dice_decimal=float(rat),
        strictly_preferred_keys=best,
        positive=total,
        centers=int(n.sum()),
    )


def run(cache, out):
    d = read(E / "predictions.json")
    ident = identity()
    assert (
        d["pipeline_id"] == token(ident)
        and d["identity"] == ident
        and set(d["specimens"]) == set("235")
        and d["shape_zyx"] == list(SHAPE)
        and d["origin_xyz"] == [360, 376, Z[0]]
        and sha(E / "counts.npz") == d["counts"]
    )
    models = {}
    with np.load(E / "counts.npz", allow_pickle=False) as f:
        for pre in ("r", "e"):
            models[pre] = (f[pre + "key"], 2 * f[pre + "positive"] - f[pre + "n"])
    allm = {
        i: {k: read_mask(E, i, k, d["specimens"][str(i)]["masks"][k]) for k in ARMS} for i in DEV
    }
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for i in DEV:
        r = d["specimens"][str(i)]
        raw, rs = verified(cache, i, "raw", False)
        assert rs["sha256"] == r["raw"]
        with Denoise(PTX / "search-support.ptx") as g:
            x = np.concatenate([graykeys(raw, i, z, g) for z in range(*Z, 4)])
        assert digest(x) == r["transfer_tuples"]
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        truth = stack(seg, i, "segmented", *Z)
        tx = label(truth)
        k, n, pos = census(x, truth)
        # Independent counting via sorted joint (tuple,phase) integer histogram.
        pair, c = np.unique(x.ravel().astype("i8") * 2 + truth.ravel(), return_counts=True)
        ix = np.searchsorted(k, pair // 2)
        p0 = np.zeros(len(k), "i8")
        p1 = np.zeros_like(p0)
        fg = (pair % 2) == 1
        p0[ix[~fg]] = c[~fg]
        p1[ix[fg]] = c[fg]
        assert np.array_equal(p1, pos) and np.array_equal(p1 + p0, n)
        B = bound(n, pos)
        path = out / f"sample{i}-target-groups.npz"
        np.savez_compressed(path, key=k, n=n, positive=pos)
        rk, prior = decode(k)
        relative = lookup(rk, prior, *models["r"])
        exact = lookup(k, relative, *models["e"])
        arms = {s: measure(v, truth, tx, i, out, s) for s, v in allm[i].items()}
        for s, m in zip(ARMS, (prior, relative, exact), strict=True):
            a = arms[s]["metrics"]
            tp = int(pos[m].sum())
            pred = int(n[m].sum())
            assert (
                a["tp"] == tp
                and a["fp"] == pred - tp
                and a["fn"] == int(pos.sum()) - tp
                and (a["fp"] + a["fn"]) >= B["unavoidable_voxel_errors"]
                and Fraction(2 * tp, pred + int(pos.sum())) <= Fraction(B["maximum_dice"])
            )
        comp = {}
        for s in ("source", "relative"):
            v = comparison(arms[s], arms["exact"])
            base, new = [arms[k]["objects"]["p95_abs_volume_error_pct"] for k in (s, "exact")]
            v["criteria"].update(
                no_lost=not v["lost"],
                all_p95=None not in (base, new) and new <= base,
                replay=not any(h["cpu_gpu_differences"] for h in r["masks"].values()),
            )
            v["advance"] = all(v["criteria"].values())
            comp[s] = v
        summary[str(i)] = all(v["advance"] for v in comp.values())
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=d["pipeline_id"],
                reference_digest=digest(truth),
                target_groups_sha256=sha(path),
                bound=B,
                arms=arms,
                comparisons=comp,
                advance=summary[str(i)],
            ),
        )
        print(i, summary[str(i)], B, flush=True)
    write(
        out / "summary.json",
        dict(pipeline_id=d["pipeline_id"], specimens=summary, advance=all(summary.values())),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
