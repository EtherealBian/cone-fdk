from itertools import product
from pathlib import Path

import numpy as np
import pytest
from pore_byte_radius import bounds, exact, possibilities


def direct(v, ct):
    mn, mx = min(v), max(v)
    m = (mn + mx) // 2
    return m < 128 if mx - mn < ct else v[0] < m


@pytest.mark.parametrize(
    "v,ct",
    [
        ((128, 128, 128, 128), 15),
        ((130, 124, 138, 122), 15),
        ((0, 0, 1, 3), 2),
        ((254, 251, 254, 255), 4),
        ((123, 121, 135, 125), 15),
        ((136, 133, 140, 127), 13),
        ((110, 95, 125, 120), 15),
    ],
)
def test_individual_patch_realizations(v, ct):
    for e in (0, 1, 2, 3):
        found = {
            direct(perturbed, ct)
            for perturbed in product(*(range(max(0, q - e), min(255, q + e) + 1) for q in v))
        }
        p, s = possibilities(v[0], min(v[1:]), max(v[1:]), ct, e)
        assert found == ({True} if p else set()) | ({False} if s else set())
        r = exact(v[0], min(v[1:]), max(v[1:]), ct)
        assert (r > 0) == direct(v, ct) and (abs(r) <= e) == (len(found) == 2)


def test_optimistic_score_dominates_all_joint_assignments():
    codes = np.array([3, -1, 2, -9, 1], dtype="i1")
    t = np.array([0, 1, 0, 1, 1], dtype=bool)
    r = bounds(codes, t, 1)
    assert (r["unavoidable_fp"], r["unavoidable_fn"]) == (2, 1)
    from pore_patch_stats import score

    for bits in product((False, True), repeat=2):
        mask = codes > 0
        mask[np.abs(codes) <= 1] = bits
        z = score(mask, t)
        assert all(z[k] <= r[k + "_upper"] for k in ("dice", "precision", "recall"))


def test_cuda_patch_radii_against_full_extrema_enumeration():
    import os

    from pore_patch_stats import Denoise

    path = Path(__file__).resolve().parents[1] / "build/pores/bernsen-radius.ptx"
    if os.name != "nt" or not path.exists():
        pytest.skip("Build registered CUDA radius PTX for runtime replay")
    rng = np.random.default_rng(919)
    im = rng.integers(115, 140, (2, 19, 23)).astype("u1")
    im[1, :9, :] = 255
    im[1, 9:, :] = 0
    circle = [(y, x) for y, x in product(range(-5, 6), repeat=2) if 0 < y * y + x * x <= 26]
    with Denoise(path) as f:
        for ct in (2, 13, 15, 21):
            a = f(im, ct)
            for z, y, x in product(range(2), range(5, 14, 2), range(5, 18, 3)):
                nb = [int(im[z, y + dy, x + dx]) for dy, dx in circle]
                assert a[z, y, x] == exact(int(im[z, y, x]), min(nb), max(nb), ct)
