import numpy as np
import pytest
from pore_five_plane import context


def test_true_axial_layout_and_shared_slice_control():
    rng = np.random.default_rng(53)
    v = rng.normal(size=(5, 6, 9, 11)).astype("f4")
    k = rng.normal(size=(2, 5, 6, 3, 3))
    out = context(v, "volume")
    ctrl = context(v, "slice")
    np.testing.assert_array_equal(out[:6], v[2])
    np.testing.assert_array_equal(ctrl[:6], out[:6])
    assert not ctrl[6:].any()
    assert not context(np.repeat(v[2:3], 5, axis=0), "volume")[6:].any()
    # A folded 2D kernel equals an explicit convolution on the original five
    # undifferenced planes, after absorbing the subtraction into the center.
    physical = np.empty_like(k)
    physical[:, [0, 1, 3, 4]] = k[:, 1:]
    physical[:, 2] = k[:, 0] - k[:, 1:].sum(axis=1)
    win = np.lib.stride_tricks.sliding_window_view
    planar = np.einsum("ocij,cyxij->oyx", k.reshape(2, 30, 3, 3), win(out, (3, 3), axis=(-2, -1)))
    volumetric = np.einsum("ozcij,zcyxij->oyx", physical, win(v, (3, 3), axis=(-2, -1)))
    np.testing.assert_allclose(planar, volumetric, atol=4e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        context(v[:4], "slice")


def test_five_plane_sealed_ids_counts_and_decisions():
    import csv
    import json

    from pore_axial_score import read_mask
    from pore_five_plane import EVID, E, sha
    from validate import final_gates

    load = lambda f: json.loads(f.read_bytes())  # noqa: E731
    seal = load(E / "predictions.json")
    prior = load(EVID / "predictions.json")
    assert set(seal["specimens"]) == set("235")
    assert seal["fits"]["slice"]["init"] == seal["fits"]["volume"]["init"]
    for k, fit in seal["fits"].items():
        assert sha(E / f"{k}.npz") == fit["weights"] and fit["trace"][-1][0] == 1535
    for i in (2, 3, 5):
        r = seal["specimens"][str(i)]
        t = load(E / f"sample{i}.json")
        old = load(EVID / f"sample{i}.json")
        assert (t["origin_xyz"], t["shape_zyx"], t["reference_mask"]) == (
            [360, 376, 256],
            [96, 256, 256],
            old["reference_mask"],
        )
        assert t["pipeline_id"] == seal["pipeline_id"] and not t["advance"]
        vals = {}
        for k, a in t["arms"].items():
            base = k == "census"
            rec = (prior["specimens"][str(i)] if base else r)["masks"][k]
            mask = read_mask(EVID if base else E, i, k, rec)
            if base:
                assert r["census_mask"] == rec["mask"]
                assert a["table_sha256"] == old["arms"][k]["table_sha256"]
            assert (
                rec["mask"] == a["mask_u8_sha256"] and rec["cpu_gpu_decision_disagreements"] == 0
            )
            m, o = a["metrics"], a["objects"]
            assert int(mask.sum()) == m["tp"] + m["fp"] == a["native"]["pore_voxels"]
            assert a["slab_depths"] == [8, 37] and a["gates_on_research_slab"] == final_gates(
                m, o
            )
            path = E / f"sample{i}-{k}-pores.csv"
            assert sha(path) == a["table_sha256"]
            with path.open(newline="") as f:
                table = {int(g["id"]): g for g in csv.DictReader(f)}
            assert (
                len(table) == a["exact_components"]
                and sum(int(g["faces"]) == 0 and int(g["voxels"]) >= 64 for g in table.values())
                == o["eligible_prediction"]
            )
            with (E / f"sample{i}-{k}-matches.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            v = {}
            for g in rows:
                n, p = int(g["ref_voxels"]), int(g["pred_voxels"])
                j = int(g["ref_scipy_label"])
                assert int(table[int(g["pred_id"])]["voxels"]) == p and float(g["iou"]) >= 0.5
                assert (
                    a["references"][str(j)]["reference_voxels"] == n
                    and a["references"][str(j)]["voxels"] == p
                )
                v[j] = abs(p - n) * 100 / n
            assert len(v) == len(rows) == o["matches"] and o["eligible_reference"] == len(
                a["references"]
            )
            assert (
                len(v) / o["eligible_reference"] == o["matched_fraction_reference"]
                and len(v) / o["eligible_prediction"] == o["matched_fraction_prediction"]
            )
            assert np.percentile(list(v.values()), 95) == pytest.approx(
                o["p95_abs_volume_error_pct"]
            )
            vals[k] = v
        for k, s in t["screens"].items():
            ids = sorted(vals[k].keys() & vals["volume"].keys())
            assert ids == s["shared_ids"]
            tails = [np.percentile([vals[a][j] for j in ids], 95) for a in (k, "volume")]
            np.testing.assert_allclose(tails, s["shared_p95"], rtol=0, atol=1e-12)
            b, a = t["arms"][k], t["arms"]["volume"]
            c = dict(
                support=len(ids) >= 5,
                p95_gain=tails[0] - tails[1] >= 2,
                dice=a["metrics"]["dice"] >= b["metrics"]["dice"] - 0.002,
                decision_replay=True,
            )
            for q in ("matched_fraction_prediction", "matched_fraction_reference"):
                c[q] = a["objects"][q] >= b["objects"][q]
            assert c == s["criteria"] and all(c.values()) == s["advance"] == (
                i == 2 and k == "census"
            )
        gains, losses = (
            vals["volume"].keys() - vals["census"].keys(),
            vals["census"].keys() - vals["volume"].keys(),
        )
        assert (gains, losses) == {
            2: ({63, 111}, set()),
            3: ({139, 184}, {137, 196}),
            5: (set(), set()),
        }[i]
