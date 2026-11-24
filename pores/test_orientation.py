from itertools import product

import numpy as np
import pytest
from pore_orientation import orient
from pore_orientation_score import ceiling


def test_eight_distinct_views_and_exact_inverse_on_rectangular_data():
    x = np.arange(2 * 3 * 11 * 15, dtype="i4").reshape(2, 3, 11, 15)
    signatures = []
    for k in range(8):
        v = orient(x, k)
        assert np.array_equal(orient(v, k, True), x)
        # Coordinate transport must commute with removal of a symmetric halo.
        assert np.array_equal(orient(x[..., 2:-2, 2:-2], k), v[..., 2:-2, 2:-2])
        assert (v[:, 0, 0, 0] - v[0, 0, 0, 0]).tolist() == [0, 3 * 11 * 15]
        signatures.append((v.shape, v.tobytes()))
    assert len(set(signatures)) == 8
    with pytest.raises(ValueError):
        orient(x, 8)


def test_group_mean_equivariant_for_deliberately_asymmetric_decoder():
    x = np.arange(10 * 13).reshape(10, 13) ** 2

    def avg(v):
        terms = []
        for k in range(8):
            a = orient(v, k)
            asymmetric = a[1:-1, 2:] - 3 * a[:-2, 1:-1]
            terms.append(orient(asymmetric, k, True))
        return np.sum(terms, axis=0)

    for k in range(8):
        assert np.array_equal(orient(avg(orient(x, k)), k, True), avg(x))


def test_oracle_dice_is_best_feasible_value_not_a_prediction():
    votes = np.array([0, 0, 1, 2, 3, 8, 8], dtype="u1")
    truth = np.array([0, 1, 1, 0, 1, 0, 1], bool)
    scores = []
    for values in product((False, True), repeat=3):
        v = votes == 8
        v[2:5] = values
        scores.append(2 * np.sum(v & truth) / (v.sum() + truth.sum()))
    result = ceiling(votes, truth)
    assert (
        result["unanimous_fn"] == 1
        and result["unanimous_fp"] == 1
        and result["dice_ceiling"] == max(scores)
    )
    assert ceiling(np.zeros(4, "u1"), np.zeros(4, bool))["dice_ceiling"] is None


@pytest.mark.parametrize("i", (2, 3, 5))
def test_archive_orientation_counts_votes_edits_and_pairs(i):
    import csv
    import json

    from pore_axial_raw import digest, sha
    from pore_axial_score import read_mask
    from pore_census_train import EVID
    from pore_orientation import ARMS, E

    def read(p):
        return json.loads(p.read_bytes())

    seal = read(E / "predictions.json")
    summary = read(E / "summary.json")
    d = read(E / f"sample{i}.json")
    r = seal["specimens"][str(i)]
    assert seal["pipeline_id"] == summary["pipeline_id"] == d["pipeline_id"]
    assert summary["advance"] == all(summary["specimens"].values())
    assert (
        d["screen"]["advance"]
        == summary["specimens"][str(i)]
        == all(d["screen"]["criteria"].values())
    )
    vp = E / f"sample{i}-votes.npz"
    assert sha(vp) == r["votes_file"]
    with np.load(vp, allow_pickle=False) as f:
        v = f["votes"]
    assert digest(v) == r["votes_u8"] and v.dtype == np.uint8
    masks = [read_mask(E, i, k, r["masks"][k]) for k in ARMS]
    old, new = masks
    assert v.shape == old.shape and np.all(new[v == 8]) and not np.any(new[v == 0])
    h = d["votes"]
    fg, bg = (np.array(h[k]) for k in ("reference_pore", "reference_material"))
    assert np.array_equal(fg + bg, np.bincount(v.ravel(), minlength=9))
    assert (fg + bg).tolist() == r["vote_counts"]
    assert (fg[0], bg[8]) == (h["unanimous_fn"], h["unanimous_fp"])
    tp = fg[1:].sum()
    assert h["dice_ceiling"] == 2 * tp / (2 * tp + fg[0] + bg[8])
    e = d["edits"]
    assert e["new_fp"] + e["fixed_fn"] == np.sum(new & ~old) == r["masks"]["mean8"]["additions"]
    assert e["fixed_fp"] + e["new_fn"] == np.sum(~new & old) == r["masks"]["mean8"]["deletions"]
    assert r["identity_parent_differences"] == 0 and len(r["view_cpu_gpu_differences"]) == 8
    assert r["masks"]["identity"]["cpu_gpu_differences"] == r["view_cpu_gpu_differences"][0]
    cp = read(EVID / f"sample{i}.json")["arms"]["census"]
    assert (
        cp["mask_u8_sha256"] == digest(old) and cp["metrics"] == d["arms"]["identity"]["metrics"]
    )
    paired = []
    for k, mask in zip(ARMS, masks, strict=True):
        a = d["arms"][k]
        tab = E / f"sample{i}-{k}-pores.csv"
        assert sha(tab) == a["table_sha256"] and a["mask_u8_sha256"] == digest(mask)
        assert a["metrics"]["tp"] + a["metrics"]["fn"] == fg.sum()
        assert len(a["gates_on_research_slab"]) == 11 and a["slab_depths"] == [8, 37]
        with tab.open(newline="") as fp:
            assert sum(1 for _ in csv.DictReader(fp)) == a["exact_components"]
        with (E / f"sample{i}-{k}-matches.csv").open(newline="") as fp:
            rows = {
                int(t["ref_scipy_label"]): abs(float(t["signed_volume_error_pct"]))
                for t in csv.DictReader(fp)
            }
        assert len(rows) == a["objects"]["matches"]
        paired.append(rows)
    a, b = paired
    ids = sorted(a.keys() & b.keys())
    assert ids == sorted(d["screen"]["shared_ids"])
    assert sorted(a.keys() - b.keys()) == d["screen"]["lost"]
    assert sorted(b.keys() - a.keys()) == d["screen"]["gained"]
    assert [np.percentile([x[j] for j in ids], 95) for x in paired] == pytest.approx(
        d["screen"]["shared_p95"]
    )
