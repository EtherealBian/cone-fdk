import hashlib
import json

import numpy as np
import pytest
from pore_axial_raw import ROOT
from pore_figure_locator import SHIFT, N, corr, reference
from pore_local_connectivity import critical, scalar

E = ROOT / "pores/evidence"


def get(p):
    return json.loads(p.read_text())


def test_masked_fft_shift_scores_against_pearson():
    rng = np.random.default_rng(794)
    a, b = rng.normal(size=(2, N, N))
    mask = rng.random((N, N)) < 0.32
    got = corr(a, *reference(b, mask))
    for dy in (-SHIFT, -2, 0, 5, SHIFT):
        for dx in (-SHIFT, -3, 0, 4, SHIFT):
            truth = np.corrcoef(np.roll(a, (-dy, -dx), (0, 1))[mask], b[mask])[0, 1]
            assert got[dy + SHIFT, dx + SHIFT] == pytest.approx(truth, abs=2e-14)


def test_locator_and_published_gray_ledger():
    loc = E / "figure-locator"
    p = E / "figure-gray"
    r = get(loc / "registration.json")
    locator = get(loc / "locator.json")
    d = get(p / "predictions.json")
    ck = get(p / "check.json")
    for f, h in [
        (p / "fields.npz", d["file"]),
        (loc / "registration.json", d["registration"]),
        (loc / "locator.json", r["locator_sha256"]),
    ]:
        assert hashlib.sha256(f.read_bytes()).hexdigest() == h
    with np.load(p / "fields.npz", allow_pickle=False) as z:
        assert {k: hashlib.sha256(z[k].tobytes()).hexdigest() for k in z.files} == d["fields"]
    assert len(locator["all_planes"]) == 900 and locator["advance"] and r["advance"]
    peak = max(locator["all_planes"], key=lambda row: row["ncc"])
    rival = max(
        (v for v in locator["all_planes"] if abs(v["z"] - peak["z"]) > 3),
        key=lambda row: row["ncc"],
    )
    assert (
        peak == locator["peak"]
        and rival == locator["rival"]
        and r["z"] == d["z"] == peak["z"] == 447
    )
    assert all(v[0] >= 616 for v in d["raw_support_xyxy"].values())
    for part, total in [("fit", 344 * 104), ("check", 344 * 196)]:
        methods = ck["partitions"][part]["methods"]
        assert set(methods) == set(d["fields"])
        for v in methods.values():
            a, w, s = [v[q] for q in ("all", "weak", "strong")]
            assert a["count"] == total == w["count"] + s["count"]
            assert a["change_rmse"] ** 2 == pytest.approx(
                sum(g["count"] * g["change_rmse"] ** 2 for g in (w, s)) / total
            )
    fit = ck["partitions"]["fit"]["methods"]
    check = ck["partitions"]["check"]["methods"]
    win = min((k for k in fit if k.startswith("r")), key=lambda k: fit[k]["all"]["change_rmse"])
    assert win == ck["selected"]
    ratios = {
        k: check[win]["all"]["change_rmse"] / check[k]["all"]["change_rmse"]
        for k in ("median", "r10h30")
    }
    assert ratios == ck["check_ratios"] and not ck["advance"] and max(ratios.values()) > 0.9
    audit = get(p / "audit.json")
    assert audit["measurements_checked"] == 420 and audit["maximum_absolute_difference"] < 2e-10


def test_local_6_connected_signed_centers_against_scalar():
    rng = np.random.default_rng(16)
    for density in (0.12, 0.44, 0.83):
        t = rng.random((7, 9, 8)) < density
        m = critical(t)
        truth = np.zeros_like(t)
        for i, j, k in np.ndindex(5, 7, 6):
            truth[i + 1, j + 1, k + 1] = scalar(t[i : i + 3, j : j + 3, k : k + 3])
        np.testing.assert_array_equal(m, truth)
    a = np.zeros((3, 3, 3), bool)
    a[1, 1, :] = True
    for bit in (False, True):
        a[1, 1, 1] = bit
        assert critical(a)[1, 1, 1]
    a[1, 0, :] = True
    assert not critical(a)[1, 1, 1]


def test_local_eligibility_by_signed_z_counts():
    d = get(E / "local-connectivity/census.json")
    assert set(d["specimens"]) == set("235")
    for row in d["specimens"].values():
        for v in row["signs"].values():
            assert len(v["per_z"]) == 24 and v["per_z"][0] == v["per_z"][-1] == 0
            assert (
                sum(v["per_z"]) == v["count"] >= 32
                and sum(x > 0 for x in v["per_z"]) == v["planes"] >= 4
            )
