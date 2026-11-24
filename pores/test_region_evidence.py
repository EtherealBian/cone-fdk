"""Archive regressions for actual regional choices, optimistic bound and persistent misses."""

import csv
from fractions import Fraction

import numpy as np
import pytest
from pore_axial_raw import digest, sha
from pore_axial_score import read_mask
from pore_object_catalog import exact, p95
from pore_persistent_miss import overlaps
from pore_region import EV, PANEL, E, read
from pore_region_headroom import optimize
from pore_transfer import label


def native_rows(folder, i, k, arm):
    p = folder / f"sample{i}-{k}-pores.csv"
    assert sha(p) == arm["table_sha256"] and arm["slab_depths"] == [8, 37]
    with p.open(newline="") as f:
        rows = {int(r["id"]): int(r["voxels"]) for r in csv.DictReader(f)}
    with (folder / f"sample{i}-{k}-matches.csv").open(newline="") as f:
        hits = {
            int(r["ref_scipy_label"]): (
                int(r["pred_id"]),
                int(r["pred_voxels"]),
                int(r["ref_voxels"]),
            )
            for r in csv.DictReader(f)
        }
    assert (
        len(rows) == arm["exact_components"]
        and len(hits) == arm["objects"]["matches"]
        and len(arm["gates_on_research_slab"]) == 11
    )
    assert all(
        rows[idx] == v
        and arm["references"][str(j)]["voxels"] == v
        and arm["references"][str(j)]["reference_voxels"] == ref
        for j, (idx, v, ref) in hits.items()
    )
    return hits


@pytest.mark.parametrize("i", (2, 3, 5))
def test_region_choice_ledger_and_actual_score(i):
    r = read(E / "predictions.json")["specimens"][str(i)]
    p = E / f"sample{i}-regions.json"
    assert (
        sha(p) == r["ledger_sha256"] and sha(E / f"sample{i}-union-pores.csv") == r["union_table"]
    )
    g = read(p)
    assert g["panel"] == [list(a) for a in PANEL] and len(g["regions"]) == r["regions"]
    med = major = 0
    for j, row in enumerate(g["regions"], 1):
        counts = {int(k): v for k, v in row["nonempty_code_counts"].items()}
        assert row["region"] == j and 0 not in counts and row["area"] == sum(counts.values())
        s = []
        for a in range(7):
            assert row["donor_sizes"][a] == sum(v for k, v in counts.items() if k & (1 << a))
            score = Fraction(0)
            for b in range(7):
                mask = (1 << a) | (1 << b)
                n = sum(v for k, v in counts.items() if k & mask)
                m = sum(v for k, v in counts.items() if k & mask == mask)
                score += Fraction(m, n) if n else 1
            s.append(score)
        assert (
            [str(v) for v in s] == row["jaccard_sums"]
            and row["tied"] == [a for a, v in enumerate(s) if v == max(s)]
            and row["chosen"] == s.index(max(s))
        )
        med += row["donor_sizes"][row["chosen"]]
        major += sum(v for k, v in counts.items() if k.bit_count() >= 4)
    real = read(E / f"sample{i}.json")
    ms = {k: read_mask(E, i, k, v) for k, v in r["masks"].items()}
    assert (
        int(ms["majority"].sum()) == major
        and int(ms["medoid"].sum()) == med
        and np.count_nonzero(ms["majority"] != ms["medoid"]) == r["majority_medoid_differences"]
    )
    for k, arm in real["arms"].items():
        native_rows(E, i, k, arm)
        if k in ms:
            assert digest(ms[k]) == arm["mask_u8_sha256"]
    assert not real["advance"] and {k: v["advance"] for k, v in real["comparisons"].items()} == {
        "census": False,
        "majority": i == 3,
    }


@pytest.mark.parametrize("i,maximum,witness", [(2, 48, 45), (3, 23, 22), (5, 10, 10)])
def test_headroom_replays_the_complete_region_choices(i, maximum, witness):
    folder = EV / "region-headroom"
    d = read(folder / f"sample{i}.json")
    s = d["summary"]
    ref = d["eligible_reference"]
    assert (
        s == read(folder / "summary.json")["specimens"][str(i)]
        and not d["cross_region"]
        and all(len(v) == 1 for v in d["owners"].values())
    )
    opt = d["options"]
    e = [[[exact(v, ref[j]) for j, v in a.items()] for a in g] for g in opt]
    sol, _ = optimize(e, len(ref))
    assert {str(n): str(q) for n, (q, _) in sol.items()} == d["at_count"]
    n = s["witness_count"]
    assert (
        max(sol) == s["max_coverage_count"] == maximum
        and n == witness
        and tuple(d["witness_choice"]) == sol[n][1]
    )
    actual = native_rows(folder, i, "oracle-witness", d["witness"])
    selected = {
        int(j): v for a, g in zip(d["witness_choice"], opt, strict=True) for j, v in g[a].items()
    }
    assert (
        selected == {j: v for j, (_, v, _) in actual.items()}
        and p95([exact(v, r) for _, v, r in actual.values()]) == sol[n][0]
    )
    assert s["possible"] == any(c >= s["needed"] and q <= 15 for c, (q, _) in sol.items())


@pytest.mark.parametrize("i,j", [(2, 28), (3, 70), (5, 62)])
def test_persistent_denominators_and_exact_overlaps(i, j):
    folder = EV / "persistent-miss"
    d = read(folder / f"sample{i}.json")
    cf = EV / "object-catalog" / f"sample{i}.json"
    assert (
        d["catalog_sha256"] == sha(cf)
        and read(cf)["never_matched"] == [j]
        and sorted(d["cases"]) == [str(j)]
    )
    c = d["cases"][str(j)]
    n = c["reference_voxels"]
    assert c["inside"]["count"] == n and len(c["donors"]) == 7
    for k, row in c["donors"].items():
        ranks = []
        for o in row["components"]:
            h = o["overlap"]
            v = o["voxels"]
            q = Fraction(h, n + v - h)
            assert (
                Fraction(o["coverage"]) == Fraction(h, n)
                and Fraction(o["purity"]) == Fraction(h, v)
                and Fraction(o["iou"]) == q
            )
            ranks.append((-q, o["seed"]))
            assert (not o["eligible"]) or q < Fraction(1, 2)
        assert (
            ranks == sorted(ranks)
            and row["overlap_total"] == sum(o["overlap"] for o in row["components"]) <= n
            and row["best_seed"] == ranks[0][1]
        )


def test_overlaps_preserve_clipped_small_and_empty_and_choose_the_lower_seed():
    t = np.zeros((7, 7, 7), bool)
    t[3, 3, 1:6] = True
    m = np.zeros_like(t)
    assert overlaps(t, label(m))["best_reason"] == "no overlap"
    m[3, 3, [1, 5]] = True
    r = overlaps(t, label(m))
    assert (
        len(r["components"]) == 2
        and r["best_reason"] == "below 64"
        and r["best_seed"] == min(q["seed"] for q in r["components"])
    )
    m[3, 3, 0] = True
    assert any(q["clipped"] for q in overlaps(t, label(m))["components"])
