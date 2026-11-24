import csv
import json

import numpy as np
import pytest
from pore_axial_raw import PTX, sha
from pore_axial_score import read_mask
from pore_mass import ARMS, E, boundary, calibration, estimate
from pore_transfer import FACE
from scipy import ndimage as ndi


def test_calibration_refuses_missing_phase_and_excludes_crop_faces():
    raw = np.full((18, 24, 28), 1000, "u2")
    mask = np.zeros(raw.shape, bool)
    assert not calibration(raw, mask)["valid"]
    mask[:, :, :14] = True
    raw[mask] = 100
    raw[:3] = 65535
    p = calibration(raw, mask)
    assert (
        p["valid"]
        and p["pore"] == {"n": 12 * 18 * 8, "median_x2": 200}
        and p["material"]["median_x2"] == 2000
    )


@pytest.mark.skipif(not (PTX / "mass-terms.ptx").is_file(), reason="requires native mass kernel")
def test_signed_integral_exactly_at_gate_and_rejects_neighbor():
    raw = np.full((19, 22, 25), 50000, "u2")
    obj = np.zeros_like(raw, bool)
    obj[7:11, 9:13, 10:14] = 1
    raw[obj] = 30000
    other = np.zeros_like(obj)
    raw[7, 8, 10] = 51000
    raw[8, 8, 10] = 49000
    est = estimate(raw, obj, other, 60000)
    assert est["reason"] == "ok" and est["numerator"] == 64 * est["denominator"]
    raw[8, 8, 10] += 1
    est = estimate(raw, obj, other, 60000)
    assert (
        est["numerator"] < 64 * est["denominator"] and est["numerator"] > 63 * est["denominator"]
    )
    other[7, 3, 10] = True
    assert estimate(raw, obj, other, 60000) == {"reason": "neighbor", "foreign_sites": 1}


def test_bright_bridge_is_not_cut_and_no_forced_eligibility():
    obj = np.zeros((9, 10, 16), bool)
    obj[3:7, 3:7, 2:6] = True
    obj[3:7, 3:7, 9:13] = True
    obj[4, 4, 6:9] = True
    raw = np.zeros(obj.shape, "u2")
    raw[4, 4, 7] = 65535
    out, d = boundary(obj, raw, 0, 1)
    assert d == -(int(obj.sum()) // 4) and ndi.label(out, FACE)[1] == 1 and out[4, 4, 7]
    obj[:] = False
    obj[3:7, 3:7, 2:6] = True
    out, d = boundary(obj, raw, 125, 2)
    assert out.sum() == 63 and d == -1


def test_sealed_per_id_mass_ledger_and_full_native_tables():
    seal = json.loads((E / "predictions.json").read_bytes())
    summary = json.loads((E / "summary.json").read_bytes())
    assert not summary["advance"] and summary["pipeline_id"] == seal["pipeline_id"]
    counted = [0, 0, 0, 0]
    for i in (2, 3, 5):
        key = str(i)
        rec = seal["specimens"][key]
        exp = json.loads((E / f"sample{i}.json").read_bytes())
        p = E / f"sample{i}-estimates.json"
        assert sha(p) == rec["estimates"]
        rows = {v["seed"]: v for v in json.loads(p.read_bytes())}
        byid = {v["seed"]: v for v in exp["previous_entrants"]}
        assert set(byid) == {j for j, r in rows.items() if r["original"] < 64 <= r["prior_gray"]}
        ordered = sorted(byid, key=lambda j: (-rows[j]["original"], j))
        keep = {j for j, r in byid.items() if r["mass"]}
        assert set(ordered[: len(keep)]) == {j for j, r in byid.items() if r["size"]}
        for j, v in byid.items():
            e = rows[j]["estimate"]
            assert v["mass"] == (e["reason"] == "ok" and e["numerator"] >= 64 * e["denominator"])
            counted[0] += v["old_category"] == "below 64"
            counted[1] += v["old_category"] == "below 64" and not v["mass"]
            counted[2] += v["old_category"] == "below 64" and e["reason"] != "ok"
            counted[3] += v["old_match"] is not None and v["mass"]
        for k in ARMS:
            m = read_mask(E, i, k, rec["masks"][k])
            a = exp["arms"][k]
            c = exp["comparisons"][k]
            table = E / f"sample{i}-{k}-pores.csv"
            assert (
                a["table_sha256"] == sha(table)
                and int(m.sum()) == a["native"]["pore_voxels"]
                and a["slab_depths"] == [8, 37]
            )
            with table.open(newline="") as f:
                eligible = {
                    int(r["id"])
                    for r in csv.DictReader(f)
                    if int(r["voxels"]) >= 64 and not int(r["faces"])
                }
            with (E / f"sample{i}-{k}-matches.csv").open(newline="") as f:
                hits = {int(r["pred_id"]): int(r["ref_scipy_label"]) for r in csv.DictReader(f)}
            assert hits.keys() <= eligible and set(hits.values()) == set(c["common"]) | set(
                c["gained"]
            )
            assert (
                len(eligible) == a["objects"]["eligible_prediction"]
                and len(hits) == a["objects"]["matches"]
                and len(a["gates_on_research_slab"]) == 11
            )
    assert counted == [23, 20, 14, 2]


def test_independent_replay_accounts_for_every_sealed_decision():
    replay = json.loads((E / "replay.json").read_bytes())
    seal = json.loads((E / "predictions.json").read_bytes())
    assert (
        replay["predictions_sha256"] == sha(E / "predictions.json")
        and replay["pipeline_id"] == seal["pipeline_id"]
    )
    for i, r in replay["development"].items():
        rec = seal["specimens"][i]
        rows = json.loads((E / f"sample{i}-estimates.json").read_bytes())
        assert (
            r["mask_differences"] == dict.fromkeys(ARMS, 0)
            and r["replayed_voxels_per_arm"] == 96 * 256 * 256
            and r["verified_objects"] == len(rows)
        )
        for k in ("mass", "size"):
            assert set(r["selected_" + k]) == {q["seed"] for q in rows if q[k]}
        assert r["phase_medians_x2"] == [
            rec["calibration"][k]["median_x2"] for k in ("pore", "material")
        ]
