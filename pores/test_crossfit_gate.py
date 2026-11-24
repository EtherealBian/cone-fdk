"""Fold isolation, exact threshold and archive evidence regressions."""

import numpy as np
import pytest
from pore_crossfit_gate import DEV, donors, names, quarter, seal, training_arrays
from pore_crossfit_gate_score import flips


@pytest.mark.parametrize("i", DEV)
def test_donors_do_not_even_read_held_out_features_or_targets(i):
    class Bank:
        def __getitem__(self, key):
            assert key in DEV and key != i
            return np.full((2, 1), key), np.full(2, 10 * key)

    x, y = training_arrays(i, Bank())
    expected = sorted(set(DEV) - {i})
    assert list(donors(i)) == expected and x[:, 0].tolist() == np.repeat(expected, 2).tolist()
    np.testing.assert_array_equal(y, 10 * x[:, 0])


@pytest.mark.parametrize("i", [1, 4, 6, 7])
def test_invalid_fold_rejected_before_access(i):
    with pytest.raises(ValueError):
        training_arrays(i, None)


def test_sampled_rows_and_targets_share_original_alternating_phase():
    t = np.arange(4 * 6 * 8).reshape(4, 6, 8)
    x = np.stack([t, t + 192], axis=-1)
    expected = [
        t[z, y, u]
        for z in range(4)
        for y in range(z % 2, 6, 2)
        for u in range((z // 2) % 2, 8, 2)
    ]
    np.testing.assert_array_equal(quarter(t), expected)
    np.testing.assert_array_equal(quarter(x), np.array([expected, np.array(expected) + 192]).T)


def test_exact_crossing_equality_common_only_and_mismatched_endpoints_rejected():
    d = dict(detector=84, full=85, reference=100, selected="curvature")
    a = {"1": d, "2": dict(d, selected="face"), "unmatched": d}
    b = {"1": dict(d, selected="face"), "2": d}
    assert flips(a, b) == dict(
        curvature_to_face=1,
        face_to_curvature=1,
        into_tolerance=1,
        out_of_tolerance=1,
        outside_before=1,
        outside_after=1,
    )
    with pytest.raises(ValueError):
        flips(a, {"2": dict(d, reference=101)})


def test_seal_rejects_missing_donor_weights_and_changed_mask(tmp_path):
    a, b, out = [tmp_path / n for n in ("a", "b", "out")]
    for folder in (a, b):
        folder.mkdir()
        for n in names():
            (folder / n).write_bytes(n.encode())
    f = b / "fold3.npz"
    f.unlink()
    with pytest.raises(ValueError):
        seal(a, b, out)
    f.write_bytes(f.name.encode())
    (b / "sample5-oof.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        seal(a, b, out)
    assert not out.exists()


@pytest.mark.parametrize("i", DEV)
def test_sealed_crossfit_measurements_and_decisions_against_native_csv(i):
    from pore_axial_raw import sha
    from pore_crossfit_gate import PREV, E, correction, read, solve
    from pore_section_gate_score import decisions
    from test_region_evidence import native_rows

    report = read(E / f"sample{i}.json")
    fit = read(E / "fit.json")
    key = str(i)
    assert np.array_equal(solve(fit["rows"]), fit["weights"])
    f = fit["calibration"][key]
    audit = read(E / "calibration-contrast.json")["specimens"][key]
    assert f["donors"] == list(donors(i)) and len(set(f["donors"] + [i])) == 3
    assert f["samples"] == 2 * 24 * 128**2 and f["weights"] == sha(E / f"fold{i}.npz")
    assert (
        audit["changed"] == audit["fixed"] + audit["broken"]
        and sum(audit["held_out"].values()) - sum(audit["original"].values())
        == audit["broken"] - audit["fixed"]
    )
    assert audit["held_out"] == {k: f[k] for k in ("fp", "fn")}
    arms = report["arms"]
    base = arms["curvature"]
    for arm, v in arms.items():
        native_rows(E, i, arm, v)
    for arm, folder in [("crossfit", E), ("previous", PREV)]:
        g = read(folder / f"sample{i}-choices.json")
        assert decisions(base, arms[arm], g) == report["decisions"][arm]
        if arm == "crossfit":
            assert all(
                r["correction_1024"] == correction(r, fit["weights"]) for r in g["sections"]
            )
    old, new = (report["decisions"][k]["rows"] for k in ("previous", "crossfit"))
    assert flips(old, new) == report["changes"]["previous"]
    assert report["advance"] == all(report["comparison"]["criteria"].values())
    s = read(E / "score-replay.json")
    assert len(s["byte_identical_tables"]) == 18 and all(
        sha(E / n) == h for n, h in s["byte_identical_tables"].items()
    )
