from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from pore_byte_window import decode, encode, graykeys, lookup
from pore_byte_window_gpu import classify
from pore_byte_window_score import bound


@pytest.mark.parametrize(
    "n,p",
    [
        ([3], [0]),
        ([3], [3]),
        ([5, 10, 9, 5], [2, 4, 7, 1]),
        ([1, 2, 5, 6, 2], [0, 2, 3, 3, 1]),
        ([2, 4, 6], [1, 2, 3]),
    ],
)
def test_exact_bound_and_error_floor_against_every_assignment(n, p):
    r = bound(n, p)
    scores, errors = [], []
    for m in product((False, True), repeat=len(n)):
        called = sum(v for v, flag in zip(n, m, strict=True) if flag)
        tp = sum(v for v, flag in zip(p, m, strict=True) if flag)
        scores.append(Fraction(2 * tp, sum(p) + called) if sum(p) + called else Fraction(1))
        errors.append(sum(p) + called - 2 * tp)
    assert Fraction(r["maximum_dice"]) == max(scores)
    assert r["unavoidable_voxel_errors"] == min(errors)


def test_gpu_integer_key_edges_ties_unknown_and_flips():
    a = np.array(
        [[0, 0, 0], [255, 255, 255], [0, 1, 255], [0, 127, 255], [127, 128, 129], [0, 127, 254]],
        dtype="u1",
    )
    k = encode(10, *a.T)
    r, s = decode(k)
    assert s.tolist() == [True, False, True, False, False, False]
    assert len(np.unique(r)) >= 4
    rk = np.unique(r[:4])
    rv = np.array([(-8 if s[np.flatnonzero(r == v)[0]] else 8) for v in rk], dtype="i4")
    # Equality after adding the prior must not flip it; some relative keys are absent.
    b = lookup(r, s, rk, rv)
    assert np.array_equal(b, s)
    ek = np.sort(k[:3])
    ev = np.array([(-9 if s[np.flatnonzero(k == v)[0]] else 9) for v in ek], dtype="i4")
    e = lookup(k, b, ek, ev)
    assert np.array_equal(e, np.r_[~s[:3], s[3:]])
    assert np.array_equal(classify(k, (rk, rv, ek, ev)), s.astype("u1") + 2 * b + 4 * e)


@pytest.mark.parametrize("i,z", [(4, 160), (6, 256), (2, 181), (3, 400), (5, 608)])
def test_graykeys_rejects_unregistered_source_before_access(i, z):
    with pytest.raises(ValueError, match="registered"):
        graykeys(None, i, z, None)


@pytest.mark.parametrize("i", (2, 3, 5))
def test_published_dictionary_ceiling_via_unsorted_fractional_iteration(i):
    from pore_axial_raw import sha
    from pore_axial_score import read_mask
    from pore_byte_window import ARMS, E
    from pore_region import read
    from test_region_evidence import native_rows

    run = read(E / f"sample{i}.json")
    rep = read(E / "replay.json")
    lock = read(E / "predictions.json")
    b = run["bound"]
    data = E / f"sample{i}-target-groups.npz"
    assert (
        sha(data)
        == run["target_groups_sha256"]
        == rep["byte_identical_target_histograms"][data.name]
    )
    with np.load(data, allow_pickle=False) as table:
        n = table["n"].astype("i8")
        p = table["positive"].astype("i8")
        assert np.all(np.diff(table["key"]) > 0)
    ref = int(p.sum())
    f = Fraction(0)
    for _ in range(100):
        take = 2 * p * f.denominator > n * f.numerator
        new = Fraction(2 * int(p[take].sum()), ref + int(n[take].sum()))
        if f == new:
            break
        assert new > f
        f = new
    else:
        pytest.fail("fractional upper bound did not converge")
    assert (
        str(f) == b["maximum_dice"]
        and float(f) == b["maximum_dice_decimal"]
        and b["unavoidable_voxel_errors"] == int(np.minimum(p, n - p).sum())
    )
    assert not run["advance"] and lock["pipeline_id"] == rep["pipeline_id"] == run["pipeline_id"]
    for k in ARMS:
        v = run["arms"][k]
        m = read_mask(E, i, k, lock["specimens"][str(i)]["masks"][k])
        native_rows(E, i, k, v)
        assert int(m.sum()) == v["metrics"]["tp"] + v["metrics"]["fp"]
        for t in ("pores", "matches"):
            s = f"sample{i}-{k}-{t}.csv"
            assert sha(E / s) == rep["byte_identical_tables"][s]
