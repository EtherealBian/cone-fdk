import csv
import hashlib
import json
from fractions import Fraction

import numpy as np
import pytest
from pore_axial_raw import ROOT, SHAPE
from pore_census_train import DEV, N, augment, schedule
from pore_slice_noise import md
from pore_window_units import BANDS, GEOM, direct

E = ROOT / "pores/evidence"


def test_window_archive_calibration_and_transition_margins():
    p = E / "window-units"
    d = json.loads((p / "predictions.json").read_text())
    ck = json.loads((p / "check.json").read_text())
    prior = json.loads((E / "bernsen-decisions.json").read_text())["specimens"]
    sums = {f"{r}-{s}": 0 for r, s in GEOM}
    for i, row in d["specimens"].items():
        for g in sums:
            scores = {
                b: Fraction(2 * f["tp"], 2 * f["tp"] + f["fp"] + f["fn"])
                for b in BANDS
                for f in [row["fit"][g][str(b)]]
            }
            chosen = max(BANDS, key=scores.get)
            assert chosen == row["band"][g]
            sums[g] += scores[chosen]
        assert row["baseline_h30"] == prior[i]["masks"]["strict"]
        ckrow = ck[i]
        hist = np.array(ckrow["truth_control_wide_histogram"]).reshape(2, 2, 2)
        f = p / f"sample{i}.npz"
        assert hashlib.sha256(f.read_bytes()).hexdigest() == row["file"]
        with np.load(f, allow_pickle=False) as z:
            a, b = (z[k] for k in ("control", "wide"))
            np.testing.assert_equal(
                np.bincount((2 * a + b).ravel(), minlength=4).reshape(2, 2), hist.sum(0)
            )
            for k, m, mat in [("control", a, hist.sum(2)), ("wide", b, hist.sum(1))]:
                assert md(m) == row["masks"][k]
                assert list(mat.ravel()) == [ckrow[k][s] for s in ("tn", "fp", "fn", "tp")]
        assert not ckrow["advance"] and ckrow["wide"]["dice"] < ckrow["control"]["dice"]
    assert max(list(sums)[1:], key=sums.get) == d["alternative"]
    assert len(d["cpu_gpu"]) == 12 and max(r["max_abs_native_gray"] for r in d["cpu_gpu"]) < 0.002


def test_direct_nlm_constant_across_every_band():
    for r, s in GEOM:
        np.testing.assert_allclose(
            direct(np.full((3, 4), 4321, dtype="u2"), 0, 2, r, s, 1), 4321, rtol=0, atol=1e-9
        )


def check_archive(study, arms):
    p = E / study
    d = json.loads((p / "predictions.json").read_text())
    fit = json.loads((p / "fit.json").read_text())
    assert fit["pipeline_id"] == d["pipeline_id"]
    assert set(d["specimens"]) == set("235")
    for k in arms:
        assert (
            hashlib.sha256((p / f"{k}.npz").read_bytes()).hexdigest() == d["fits"][k]["weights"]
        )
    for i, rec in d["specimens"].items():
        res = json.loads((p / f"sample{i}.json").read_text())
        assert res["pipeline_id"] == d["pipeline_id"] and res["shape_zyx"] == list(SHAPE)
        common = None
        rel = {}
        for k in arms:
            r = res["arms"][k]
            lock = rec["masks"][k]
            f = p / f"sample{i}-{k}.npz"
            assert hashlib.sha256(f.read_bytes()).hexdigest() == lock["file"]
            assert res["screen"]["criteria"]["decision_replay"] == all(
                row["cpu_gpu_decision_disagreements"] == 0 for row in rec["masks"].values()
            )
            with np.load(f, allow_pickle=False) as z:
                m = z["mask"]
                assert (
                    m.shape == SHAPE
                    and m.dtype == bool
                    and hashlib.sha256(m.tobytes()).hexdigest()
                    == lock["mask"]
                    == r["mask_u8_sha256"]
                )
                assert m.sum() == r["metrics"]["tp"] + r["metrics"]["fp"]
            tab = p / f"sample{i}-{k}-pores.csv"
            assert hashlib.sha256(tab.read_bytes()).hexdigest() == r["table_sha256"]
            with tab.open(newline="") as fp:
                assert sum(1 for _ in csv.DictReader(fp)) == r["exact_components"]
            with (p / f"sample{i}-{k}-matches.csv").open(newline="") as fp:
                pairs = list(csv.DictReader(fp))
                assert len(pairs) == r["objects"]["matches"]
            rel[k] = {
                int(v["ref_scipy_label"]): abs(float(v["signed_volume_error_pct"])) for v in pairs
            }
            common = set(rel[k]) if common is None else common & rel[k].keys()
            for v in pairs:
                ref = r["references"][v["ref_scipy_label"]]
                assert [ref["reference_voxels"], ref["voxels"]] == [
                    int(v["ref_voxels"]),
                    int(v["pred_voxels"]),
                ]
                assert ref["iou"] == pytest.approx(float(v["iou"]))
        assert sorted(common) == sorted(res["screen"]["shared_ids"])
        np.testing.assert_allclose(
            [np.percentile([rel[k][rid] for rid in common], 95) for k in arms],
            res["screen"]["shared_p95"],
            atol=1e-11,
            rtol=0,
        )


def test_axial_archive_matches_and_local_trace():
    check_archive("axial-raw", ("plane", "axial"))
    d = json.loads((E / "axial-raw/trace-5-15.json").read_text())
    t = d["arms"]
    assert d["reference_voxels"] == 452 and not t["plane"][0]["unclipped"]
    a = t["axial"][0]
    assert (
        a["unclipped"] and (a["intersection"], a["union"]) == (394, 458) and a["iou"] == 394 / 458
    )


def test_census_geometry_and_paired_augmentations():
    p = np.array([[0, 0, 0], [23, 255, 255], [9, 128, 128]])
    rows, counts = schedule({i: (p, p) for i in DEV})
    a = np.arange(35).reshape(5, 7)
    for flag in range(8):
        np.testing.assert_equal(augment(np.stack([a, a]), flag)[0], augment(a, flag))
    for i in DEV:
        a, b = [rows[k][i] for k in ("replacement", "census")]
        np.testing.assert_equal(a[:, 1], b[:, 1])
        np.testing.assert_equal(a[:, :, 3], b[:, :, 3])
        assert len(set(map(tuple, b[:, 0, :3]))) == N
        assert (
            counts["census"][i]["unique_voxels"] == 24 * 256 * 256
            and counts["replacement"][i]["unique_voxels"] < 24 * 256 * 256
        )


def test_census_archive_and_full_coverage():
    check_archive("census-train", ("replacement", "census"))
    d = json.loads((E / "census-train/predictions.json").read_text())
    for i in ("2", "3", "5"):
        a, b = [d["coverage"][k][i] for k in ("replacement", "census")]
        assert (
            a["common_sha256"] == b["common_sha256"]
            and a["augment_sha256"] == b["augment_sha256"]
        )
        assert (
            a["unique_grid_tiles"] < b["unique_grid_tiles"] == N
            and a["unique_voxels"] < b["unique_voxels"] == 1572864
        )


def test_signed_connectivity_archive_gain_and_loss():
    check_archive("local-connectivity", ("census", "critical"))

    def get(p):
        return json.loads((E / p).read_bytes())

    d = get("local-connectivity/predictions.json")
    census = get("local-connectivity/census.json")
    old = get("census-train/predictions.json")
    expected = {
        "2": ([12, 56, 63, 87, 111], [26, 27]),
        "3": ([71], [125, 153, 186]),
        "5": ([], []),
    }
    for i, row in d["specimens"].items():
        assert row["masks"]["census"]["mask"] == old["specimens"][i]["masks"]["census"]["mask"]
        assert d["coverage"][i] == old["coverage"]["census"][i]
        res = get(f"local-connectivity/sample{i}.json")
        a, b = res["arms"].values()
        ids = [
            {int(j) for j, v in r["references"].items() if v["voxels"] is not None}
            for r in (a, b)
        ]
        assert (sorted(ids[1] - ids[0]), sorted(ids[0] - ids[1])) == expected[i]
        for key, exp in zip(("gap", "connector"), d["exposures_gap_connector"][i], strict=True):
            sign = census["specimens"][i]["signs"][key]
            assert (
                exp >= sign["count"] == sum(sign["per_z"]) >= 32
                and sign["planes"] == np.count_nonzero(sign["per_z"]) >= 4
            )
        sc = res["screen"]
        p, q = sc["shared_p95"]
        assert sc["criteria"]["p95_gain"] == (p - q >= 2)
        for k in ("matched_fraction_reference", "matched_fraction_prediction"):
            assert sc["criteria"][k] == (b["objects"][k] >= a["objects"][k])
        assert not sc["advance"] and not all(sc["criteria"].values())
    trace = get("local-connectivity/trace-5-15.json")
    assert trace["reference_voxels"] == 452
    for k, n, hit in [("census", 1902, 406), ("critical", 2001, 411)]:
        (r,) = trace["arms"][k]
        assert (r["voxels"], r["intersection"], r["unclipped"]) == (n, hit, False) and r[
            "iou"
        ] == hit / (452 + n - hit)
