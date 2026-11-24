import json
from fractions import Fraction

import numpy as np
import pytest
from pore_axial_raw import ROOT, digest, sha
from pore_median_source import ALTS, native, scalar
from pore_patch_profile import ARMS, direct
from pore_slice_noise import md
from pore_window_units import BANDS
from pore_window_units import direct as uniform_direct

E = ROOT / "pores/evidence"


def get(p):
    return json.loads((E / p).read_bytes())


def test_patch_profile_archive_and_independent_distance():
    p = E / "patch-profile"
    d = get("patch-profile/predictions.json")
    ck = get("patch-profile/check.json")
    gray = get("figure-gray/predictions.json")
    old = get("window-units/predictions.json")
    assert sha(p / "fields.npz") == d["paper"]["file"]
    with np.load(p / "fields.npz", allow_pickle=False) as z:
        assert {k: digest(z[k]) for k in z.files} == d["paper"]["fields"] and len(z.files) == 12
        for h in BANDS:
            assert digest(z[f"uniform{h}"]) == gray["fields"][f"r10h{h}"]
    im = np.random.default_rng(9621).integers(25000, 42001, (1, 23, 29), dtype="u2")[0]
    for r in d["numeric"]:
        y, x = r["y"], r["x"]
        b = r["arm"] == "binomial"
        vals = direct(im, y, x, b, 160)
        np.testing.assert_allclose(r["cpu"], vals, atol=1e-9, rtol=0)
        np.testing.assert_allclose(r["cuda"], vals, atol=0.003, rtol=0)
        if not b:
            np.testing.assert_allclose(
                vals, uniform_direct(im, y, x, 10, 2, 160), atol=1e-8, rtol=0
            )
    assert len(d["numeric"]) == 6
    for i, r in d["specimens"].items():
        rec = ck["specimens"][i]
        hist = np.array(rec["truth_uniform_binomial"]).reshape(2, 2, 2)
        fn = p / f"sample{i}.npz"
        assert sha(fn) == r["file"]
        assert r["uniform_fixed_h30"] == old["specimens"][i]["baseline_h30"]
        assert r["masks"]["uniform"] == old["specimens"][i]["masks"]["control"]
        with np.load(fn, allow_pickle=False) as z:
            a, b = [z[k] for k in ARMS]
            np.testing.assert_equal(
                np.bincount((2 * a + b).ravel(), minlength=4), hist.sum(0).ravel()
            )
            for k, m, tab in zip(ARMS, (a, b), (hist.sum(2), hist.sum(1)), strict=True):
                assert r["masks"][k] == md(m)
                assert tab.ravel().tolist() == [rec[k][cell] for cell in ("tn", "fp", "fn", "tp")]
                fits = r["fit"][k]
                frac = {
                    h: Fraction(2 * f["tp"], 2 * f["tp"] + f["fp"] + f["fn"])
                    for h in BANDS
                    for f in [fits[str(h)]]
                }
                assert max(BANDS, key=frac.get) == r["band"][k]
        assert not rec["advance"] and rec["binomial"]["dice"] < rec["uniform"]["dice"]
    pap = ck["paper"]
    fit, right = [pap["parts"][k]["methods"] for k in ("fit", "check")]
    chosen = [min(BANDS, key=lambda h: fit[f"{k}{h}"]["all"]["change_rmse"]) for k in ARMS]
    assert chosen == [pap["selected"][k] for k in ARMS]
    a, b = [right[f"{k}{h}"]["all"]["change_rmse"] for k, h in zip(ARMS, chosen, strict=True)]
    assert b / a == pytest.approx(pap["right_ratio"]) and b > 0.9 * a and not ck["advance"]


def test_rank_filtered_fields_and_partition_selection():
    p = E / "median-source"
    d = get("median-source/predictions.json")
    ck = get("median-source/check.json")
    gray = get("figure-gray/predictions.json")
    assert (
        sha(p / "fields.npz") == d["file"]
        and len(set(map(tuple, d["scalar_positions"]))) == d["verified_per_arm"] == 256
    )
    with np.load(p / "fields.npz", allow_pickle=False) as z:
        assert {k: digest(z[k]) for k in z.files} == d["fields"] and len(z.files) == 11
    assert (
        d["fields"]["cube27"] == gray["fields"]["median"]
        and d["fields"]["original"] == gray["fields"]["original"]
    )
    im = np.random.default_rng(123).integers(0, 65536, (3, 7, 8), dtype="u2")
    for k in ["cube27", *ALTS]:
        field = native(im, k)
        for y, x in ((1, 2), (3, 6), (4, 3)):
            assert field[y, x] == scalar(im[:, y - 1 : y + 2, x - 1 : x + 2], k)
    fit, right = [ck["partitions"][k] for k in ("fit", "check")]
    best = min(ALTS, key=lambda k: fit[k]["all"]["change_rmse"])
    assert best == ck["selected"]
    control = get("patch-profile/check.json")["paper"]["parts"]["check"]["median_control"][
        "native_median"
    ]
    assert right["cube27"] == control
    new = right[best]["all"]
    ref = control["all"]
    assert new["change_rmse"] / ref["change_rmse"] == ck["check_ratio"]
    assert ck["advance"] == (
        ck["check_ratio"] <= 0.9 and new["absolute_rmse"] <= ref["absolute_rmse"]
    )
