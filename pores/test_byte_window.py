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
