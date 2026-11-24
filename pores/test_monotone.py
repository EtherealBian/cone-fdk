import csv

import numpy as np
import pytest
from pore_axial_raw import PTX, sha
from pore_axial_score import read_mask
from pore_forward import CONTROL
from pore_forward_energy import energy
from pore_monotone import ARMS, OUT, color_cpu, color_gpu, load, support


@pytest.mark.parametrize("q", [CONTROL, [0, 0, 967, 1819, 1310]])
def test_colored_updates_match_brute_force_single_flip_energies(q):
    rng = np.random.default_rng(820)
    base = rng.random((11, 12, 13)) < 0.4
    raw = rng.integers(0, 65536, base.shape, dtype="u2")
    p, m = 53000, 92421
    u = (m - 2 * raw.astype("f8")) / (m - p)
    before = energy(base, base, u, q)["total"]
    brute = base.copy()
    sites = sorted(
        map(tuple, np.argwhere(support(base))), key=lambda pos: tuple(v % 5 for v in pos)
    )
    for site in sites:
        one = brute.copy()
        one[site] = not brute[site]
        if energy(one, base, u, q)["total"] < energy(brute, base, u, q)["total"]:
            brute = one
    cpu, gain, trace = color_cpu(base, raw, q, (p, m))
    assert np.array_equal(cpu, brute) and np.array_equal(gain > 0, base != cpu)
    assert len(trace) == 125 and all(r["gain_numerator"] >= 0 for r in trace)
    observed = before - energy(cpu, base, u, q)["total"]
    assert observed == pytest.approx(
        sum(r["gain_numerator"] for r in trace) / (6 * (m - p) * 98304**2 * base.size), abs=1e-13
    )
    if (PTX / "pore-monotone.ptx").is_file():
        v, g = color_gpu(base, raw, q, (p, m))
        assert np.array_equal(v, cpu) and np.array_equal(g, gain)


def test_no_boundary_no_edits_even_with_opposite_intensity():
    base = np.ones((11, 12, 13), bool)
    raw = np.full(base.shape, 65535, "u2")
    result, gain, trace = color_cpu(base, raw, CONTROL, (400, 800))
    assert not support(base).any() and not gain.any() and np.array_equal(result, base)
    assert sum(r["available"] for r in trace) == 0


def test_sealed_ledger_includes_every_color_match_and_object():
    seal = load(OUT, "predictions.json")
    summary = load(OUT, "summary.json")
    assert seal["pipeline_id"] == summary["pipeline_id"]
    for i in (2, 3, 5):
        r = seal["specimens"][str(i)]
        data = load(OUT, f"sample{i}.json")
        trace = r["colors"]
        color = r["masks"]["band-color"]
        assert [t["color"] for t in trace] == list(range(125)) and sum(
            t["available"] for t in trace
        ) == r["support"]
        assert sum(t["changed"] for t in trace) == color["additions"] + color["deletions"]
        assert r["initial_energy"]["total"] - color["energy"]["total"] == pytest.approx(
            r["per_voxel_gain"], abs=1e-10
        )
        for k in ARMS:
            m = read_mask(OUT, i, k, r["masks"][k])
            a = data["arms"][k]
            e = data["edits"][k]
            tab = OUT / f"sample{i}-{k}-pores.csv"
            assert sha(tab) == a["table_sha256"] and a["native"]["pore_voxels"] == int(m.sum())
            with tab.open(newline="") as fp:
                rows = list(csv.DictReader(fp))
            assert len(rows) == a["exact_components"] and len(a["gates_on_research_slab"]) == 11
            with (OUT / f"sample{i}-{k}-matches.csv").open(newline="") as fp:
                hits = {int(row["ref_scipy_label"]) for row in csv.DictReader(fp)}
            assert len(hits) == a["objects"]["matches"] and a["slab_depths"] == [8, 37]
            assert (
                e["new_false_positive"] + e["corrected_false_negative"]
                == r["masks"][k]["additions"]
            )
            assert (
                e["new_false_negative"] + e["corrected_false_positive"]
                == r["masks"][k]["deletions"]
            )
            assert (
                r["masks"][k]["cpu_gpu_differences"]
                == e["deep_false_positive"]
                == e["deep_false_negative"]
                == e["off_support"]
                == 0
            )
            for control, c in data["comparisons"][k].items():
                paired = set(c["shared_ids"])
                assert hits == paired | set(c["gained"])
                assert paired | set(c["lost"]) == {
                    int(j)
                    for j, v in data["arms"][control]["references"].items()
                    if v["pred"] is not None
                }
