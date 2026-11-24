"""Exact oracle for whole-region donor choices; never supplies candidates to the predictor."""

import argparse
import csv
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, Z, digest, stack, verified, write
from pore_object_catalog import exact, p95
from pore_region import EV, PANEL, E, donor, read
from pore_transfer import label, measure


def trim(values, keep):
    return tuple(sorted(values, reverse=True)[:keep])


def optimize(options, nref):
    if nref < 1:
        raise ValueError("No reference population")
    keep = nref - (19 * (nref - 1) // 20)
    states = {0: {(): ()}}
    history = []
    for opts in options:
        if not opts:
            raise ValueError("No region options")
        subset = {}
        for idx, errs in enumerate(opts):
            subset.setdefault((len(errs), trim(errs, keep)), idx)
        new = defaultdict(dict)
        for n, prev in states.items():
            for worst, choices in prev.items():
                for (m, tail), idx in subset.items():
                    if n + m > nref:
                        raise ValueError("Overlapping reference populations")
                    sig = trim(worst + tail, keep)
                    new[n + m].setdefault(sig, choices + (idx,))
        states = {}
        for n, group in new.items():
            kept = {}
            for vec in sorted(group):
                # Lexicographic pre-sort means a future distinct vector cannot dominate a prior vector.
                if not any(all(x <= y for x, y in zip(k, vec, strict=True)) for k in kept):
                    kept[vec] = group[vec]
            states[n] = kept
        history.append(sum(map(len, states.values())))
    result = {}
    for n, d in states.items():

        def tail95(desc, n=n):
            if not n:
                return Fraction(0)
            q, r = divmod(19 * (n - 1), 20)
            a = n - 1 - q
            return (desc[a] * (20 - r) + desc[max(a - 1, 0)] * r) / 20

        v = min(d, key=lambda k: (tail95(k), d[k]))
        result[n] = (tail95(v) if n else None, d[v])
    return result, history


def options_for(i, masks, regions, tx):
    lt, ct, whole, _ = tx
    rid = set(map(int, np.flatnonzero(whole & (ct >= 64))))
    count = int(regions.max())
    opts = [[{} for _ in PANEL] for _ in range(count)]
    owner = defaultdict(set)
    all_edges = []
    for a, ((study, arm), m) in enumerate(zip(PANEL, masks, strict=True)):
        lp, pc, pw, _ = label(m)
        keep = pw & (pc >= 64)
        fg = np.flatnonzero(lp)
        ids = lp.ravel()[fg]
        seeds = np.full(len(pc), m.size, "i8")
        np.minimum.at(seeds, ids, fg)
        both = keep[lp] & (lt != 0) & whole[lt] & (ct[lt] >= 64)
        keys, hits = np.unique(lp[both].astype("i8") * len(ct) + lt[both], return_counts=True)
        edges = []
        for key, h in zip(keys, hits, strict=True):
            pred, j = divmod(int(key), len(ct))
            union = int(pc[pred] + ct[j] - h)
            if 2 * h >= union:
                reg = int(regions.ravel()[seeds[pred]])
                owner[j].add(reg)
                edges.append((Fraction(int(h), union), pred, j, reg))
        used_p = set()
        used_r = set()
        observed = {}
        edge_info = []
        for iou, p, j, reg in sorted(edges, reverse=True):
            edge_info.append([reg, int(seeds[p]) + 1, j, str(iou)])
            if p in used_p or j in used_r:
                continue
            used_p.add(p)
            used_r.add(j)
            observed[j] = (int(seeds[p]) + 1, int(pc[p]), int(ct[j]))
            opts[reg - 1][a][j] = int(pc[p])
        tab = EV / study / f"sample{i}-{arm}-matches.csv"
        with tab.open(newline="") as f:
            original = {
                int(h["ref_scipy_label"]): (
                    int(h["pred_id"]),
                    int(h["pred_voxels"]),
                    int(h["ref_voxels"]),
                )
                for h in csv.DictReader(f)
            }
        if original != observed:
            raise ValueError(f"Source overlap/CSV disagreement: {study} {i}")
        all_edges.append(edge_info)
    return rid, opts, owner, all_edges


def run(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    seal = read(E / "predictions.json")
    summary = {}
    for i in DEV:
        rec = seal["specimens"][str(i)]
        ms = np.stack([donor(i, k, a)[0] for k, a in PANEL])
        assert [digest(v) for v in ms] == rec["donor_masks"]
        u = ms.any(0)
        assert digest(u) == rec["union_mask"]
        regions = label(u)[0]
        assert regions.max() == rec["regions"]
        seg, sig = verified(cache, i, "segmented", False)
        assert sig["sha256"] == rec["segmented"]
        t = stack(seg, i, "segmented", *Z)
        tx = label(t)
        assert digest(t) == read(E / f"sample{i}.json")["reference_digest"]
        rid, opts, owner, edges = options_for(i, ms, regions, tx)
        sizes = tx[1]
        multi = {int(k): sorted(v) for k, v in owner.items() if len(v) > 1}
        detail = dict(
            specimen=i,
            source_segmented=rec["segmented"],
            reference_digest=digest(t),
            panel=PANEL,
            eligible_reference={j: int(sizes[j]) for j in sorted(rid)},
            owners={int(k): sorted(v) for k, v in owner.items()},
            cross_region=multi,
            edges_by_donor=edges,
            options=opts,
            never_matched=sorted(rid - set().union(*(set(v) for group in opts for v in group))),
        )
        if multi:
            write(out / f"sample{i}.json", detail)
            summary[str(i)] = dict(separable=False)
            continue
        errs = [[[exact(v, int(sizes[j])) for j, v in h.items()] for h in g] for g in opts]
        nref = len(rid)
        k = (9 * nref + 9) // 10
        results, trace = optimize(errs, nref)
        feasible = [n for n in results if n >= k]
        n = min(feasible, key=lambda n: (results[n][0], -n)) if feasible else max(results)
        score, chosen = results[n]
        which = np.asarray((0, *chosen), "u1")
        m = np.take_along_axis(ms, which[regions][None], axis=0)[0]
        expected = {j: v for reg, a in enumerate(chosen) for j, v in opts[reg][a].items()}
        assert len(expected) == n
        witness = measure(m, t, tx, i, out, "oracle-witness")
        actual = {
            int(j): r["voxels"] for j, r in witness["references"].items() if r["pred"] is not None
        }
        assert actual == expected and witness["objects"]["matches"] == n
        tail = p95([exact(v, int(sizes[j])) for j, v in actual.items()])
        assert tail == score and (
            tail is None
            or abs(float(tail) - witness["objects"]["p95_abs_volume_error_pct"]) < 1e-9
        )
        unmixed = {}
        for a, (study, arm) in enumerate(PANEL):
            allerrs = [v for g in errs for v in g[a]]
            q = p95(allerrs)
            unmixed[study + "/" + arm] = dict(
                matched=len(allerrs), p95=str(q) if q is not None else None
            )
        summary[str(i)] = dict(
            separable=True,
            reference=nref,
            needed=k,
            max_coverage_count=max(results),
            witness_count=n,
            witness_p95=float(score) if score is not None else None,
            possible=bool(feasible) and score <= 15,
        )
        detail.update(
            summary=summary[str(i)],
            unmixed=unmixed,
            dp_state_counts=trace,
            at_count={
                n: str(q) if q is not None else None for n, (q, _) in sorted(results.items())
            },
            regions_with_multiple_reference_ids=[
                j for j, g in enumerate(opts, 1) if len(set().union(*g)) > 1
            ],
            witness_choice=chosen,
            witness=witness,
        )
        write(out / f"sample{i}.json", detail)
        print(i, summary[str(i)], "peak states", max(trace), flush=True)
    write(out / "summary.json", dict(diagnostic_only=True, specimens=summary))


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
