import csv
import json

import numpy as np
import pytest
from pore_axial_raw import PTX, sha
from pore_axial_score import read_mask
from pore_forward import CONTROL, COUNTS, D, E, kernel, native, simplex
from pore_forward_energy import energy
from pore_forward_inverse import cpu_step
from scipy import ndimage as ndi

FIT = [0, 0, 967, 1819, 1310]


def test_forward_shell_counts_simplex_and_quantization():
    assert COUNTS.tolist() == [1, 6, 12, 8, 6]
    for q in (FIT, CONTROL):
        assert kernel(q)[1].sum() == D
    target = np.array([0.0, 0.25, 0.0, 0.5, 0.25])
    a, q, _ = simplex(np.eye(5), target)
    assert np.allclose(a, target) and np.array_equal(q, 4096 * target)


@pytest.mark.parametrize("q", [FIT, CONTROL])
def test_forward_single_flip_matches_independent_objective(q):
    rng = np.random.default_rng(152)
    m = rng.random((11, 12, 13)) < 0.45
    raw = rng.integers(0, 65536, m.shape, dtype="u2")
    p, t = 53000, 92421
    u = (t - raw.astype("f8") * 2) / (t - p)
    expected = m.copy()
    e0 = energy(m, m, u, q)["total"]
    for a in np.ndindex(3, 4, 5):
        i = tuple(v + 4 for v in a)
        one = m.copy()
        one[i] = not m[i]
        if energy(one, m, u, q)["total"] < e0:
            expected[i] = one[i]
    cpu, _ = cpu_step(m, raw, q, (p, t))
    assert np.array_equal(cpu, expected)
    if (PTX / "pore-forward.ptx").is_file():
        w, g = kernel(q)
        assert np.array_equal(native(m, w), ndi.correlate(m.astype("i4"), g, mode="nearest"))
        assert np.array_equal(native(m, w, raw, (p, t)), expected)


def test_exact_ties_keep_both_classes_in_flat_fields():
    # Sharp kernel, constant neighbors: flip costs 1 (data curvature), 1/2 (cut), 1/2 (prior).
    for orig in (False, True):
        m = np.full((11, 11, 11), orig)
        raw = np.full(m.shape, 400 if orig else 200, "u2")
        out, ties = cpu_step(m, raw, CONTROL, (400, 800))
        assert np.array_equal(out, m) and ties == 27
        if (PTX / "pore-forward.ptx").is_file():
            assert np.array_equal(native(m, kernel(CONTROL)[0], raw, (400, 800)), m)


def test_forward_evidence_covers_all_objects_and_empty_matches():
    def get(name):
        return json.loads((E / name).read_bytes())

    seal, summary, check, en = map(
        get, ["predictions.json", "summary.json", "check.json", "energy.json"]
    )
    assert check["passed"] and not summary["advance"]
    assert en["posthoc"] and en["predictions_sha256"] == sha(E / "predictions.json")
    assert seal["pipeline_id"] == summary["pipeline_id"]
    for i in (2, 3, 5):
        key = str(i)
        data = get(f"sample{i}.json")
        assert data["pipeline_id"] == seal["pipeline_id"] and all(
            check["specimens"][key]["criteria"].values()
        )
        for k, a in data["arms"].items():
            table = E / f"sample{i}-{k}-pores.csv"
            assert a["table_sha256"] == sha(table) and a["slab_depths"] == [8, 37]
            with table.open(newline="") as f:
                rows = list(csv.DictReader(f))
            eligible = {
                int(r["id"]) for r in rows if int(r["voxels"]) >= 64 and not int(r["faces"])
            }
            with (E / f"sample{i}-{k}-matches.csv").open(newline="") as f:
                matched = list(csv.DictReader(f))
            assert len(rows) == a["exact_components"] == a["native"]["components"]
            assert (
                len(eligible) == a["objects"]["eligible_prediction"]
                and len(a["gates_on_research_slab"]) == 11
            )
            assert {int(r["pred_id"]) for r in matched} <= eligible and len(matched) == a[
                "objects"
            ]["matches"]
            if k != "census":
                r = seal["specimens"][key]["masks"][k]
                m = read_mask(E, i, k, r)
                assert r["cpu_gpu_differences"] == 0 and m.sum() == a["native"]["pore_voxels"]
                assert (
                    en["specimens"][key][k]["after"]["changed"] == r["additions"] + r["deletions"]
                )
        assert not data["arms"]["blur"]["objects"]["matches"] and not any(
            data["arms"]["blur"]["gates_on_research_slab"].values()
        )
        for k, c in data["screens"].items():
            assert not c["advance"] and not c["gained"] and c["shared_p95"] == [None, None]
            assert set(c["lost"]) == {
                int(j) for j, v in data["arms"][k]["references"].items() if v["pred"] is not None
            }
