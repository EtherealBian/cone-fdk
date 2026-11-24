import csv

import numpy as np
import pytest
from pore_axial_raw import digest, sha
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_distance import ARMS, E, load
from pore_distance_targets import describe, targets


def test_distances_agree_with_brute_force_on_anisymmetric_binary_crop():
    rng = np.random.default_rng(7102)
    t = rng.random((16, 18, 15)) < 0.05
    groups = {k: np.argwhere(t == k) for k in (False, True)}
    d = targets(t)
    assert d.dtype == np.float32
    expected = np.zeros_like(d)
    for z, y, x in np.ndindex(*(a - 8 for a in t.shape)):
        pos = np.array([z, y, x]) + 4
        site = tuple(pos)
        k = bool(t[site])
        other = groups[not k]
        nearest = np.sqrt(np.min(np.sum((other - pos) ** 2, axis=1)))
        if nearest <= 4:
            expected[site] = (1 if k else -1) * (nearest - 0.5)
    assert np.array_equal(d, expected) and np.array_equal(targets(~t), -expected)
    assert np.array_equal(targets(t[::-1].swapaxes(0, 2)), d[::-1].swapaxes(0, 2))


def test_edge_exterior_is_not_a_target_and_empty_classes_never_invent_support():
    t = np.ones((17, 15, 17), bool)
    assert not targets(t).any() and not targets(~t).any()
    t[:, :, :8] = False
    d = targets(t)
    v = d[8, 7]
    assert np.array_equal(v[4:12], [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
    assert not v[:4].any() and not v[12:].any()
    r = describe(d)
    assert r["sides"]["pore"]["bins"] == r["sides"]["material"]["bins"]
    with pytest.raises(ValueError):
        targets(np.zeros((8, 9, 9), bool))


@pytest.mark.parametrize("i", (2, 3, 5))
def test_public_distance_census_fits_masks_and_object_ledgers(i):
    d = load(E / f"sample{i}.json")
    seal = load(E / "predictions.json")
    summ = load(E / "summary.json")
    parent = load(EVID / "predictions.json")
    r = seal["specimens"][str(i)]
    c = load(E / "census.json")["specimens"][str(i)]
    assert c["eligible"] and c["cal_mask"] == r["cal_mask"] and c["segmented"] == r["segmented"]
    for s in c["sides"].values():
        assert sum(s["bins"]) == s["centers"] and sum(s["beyond_one_by_z"]) == s[
            "beyond_one"
        ] == sum(s["bins"][1:])
    for name in ("init", "stream", "band_exposures", "empty_batches"):
        assert seal["fits"]["sign"][name] == seal["fits"]["distance"][name]
    assert d["pipeline_id"] == summ["pipeline_id"] == seal["pipeline_id"] and d[
        "advance"
    ] == summ["specimens"][str(i)] == all(c["advance"] for c in d["comparisons"].values())
    assert summ["advance"] == all(summ["specimens"].values())
    hits = {}
    for k in (*ARMS, "census"):
        mask = read_mask(
            EVID if k == "census" else E,
            i,
            k,
            parent["specimens"][str(i)]["masks"][k] if k == "census" else r["masks"][k],
        )
        a = d["arms"][k]
        if k in ARMS:
            assert sha(E / f"{k}.npz") == seal["fits"][k]["weights"]
            assert (r["masks"][k]["first_difference"] is None) == (
                r["masks"][k]["cpu_gpu_differences"] == 0
            )
        assert (
            a["native"]["pore_voxels"] == int(mask.sum()) and digest(mask) == a["mask_u8_sha256"]
        )
        assert len(a["gates_on_research_slab"]) == 11 and a["slab_depths"] == [8, 37]
        tab = E / f"sample{i}-{k}-pores.csv"
        assert sha(tab) == a["table_sha256"]
        with tab.open(newline="") as f:
            assert sum(1 for _ in csv.DictReader(f)) == a["exact_components"]
        with (E / f"sample{i}-{k}-matches.csv").open(newline="") as f:
            rows = {
                int(v["ref_scipy_label"]): abs(float(v["signed_volume_error_pct"]))
                for v in csv.DictReader(f)
            }
        assert len(rows) == a["objects"]["matches"]
        hits[k] = rows
        assert a["objects"]["p95_abs_volume_error_pct"] == pytest.approx(
            np.percentile(list(rows.values()), 95)
        )
    for k, comp in d["comparisons"].items():
        aa, bb = hits[k], hits["distance"]
        ids = sorted(aa.keys() & bb.keys())
        assert (
            ids == sorted(comp["shared_ids"])
            and sorted(aa.keys() - bb.keys()) == comp["lost"]
            and sorted(bb.keys() - aa.keys()) == comp["gained"]
        )
        assert comp["shared_p95"] == pytest.approx(
            [np.percentile([row[j] for j in ids], 95) for row in (aa, bb)]
        ) and comp["advance"] == all(comp["criteria"].values())
