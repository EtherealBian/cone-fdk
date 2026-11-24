import numpy as np
import pytest
from pore_inverse_weight import ROOT, KernelStats, blend, run, search
from pore_public_filter import controls, describe
from scipy import ndimage as ndi


def test_inverse_stats_scalar_replay_and_normalization():
    a = np.random.default_rng(943).integers(0, 65536, (2, 17, 20), dtype="u2")
    a[1] = 65535
    p = np.pad(a.astype("f8"), ((0, 0), (12, 12), (12, 12)), mode="edge")
    co = np.array([0.25, 0, 0, 0, 0.25, 0, 0.5])
    h = 100.0
    with KernelStats(ROOT / "build/pores/inverse-weight.ptx") as gpu:
        with pytest.raises(ValueError):
            gpu(a.astype("f4"), h)
        stat = gpu(a, h)
    img = blend(stat, co)
    # The published ABI stores the *sum*, not its normalized mean, in binary32.
    exact = 65535 * 441
    stored = np.float32(exact)
    np.testing.assert_array_equal(stat[1, :7], stored)
    np.testing.assert_array_equal(stat[1, 7:], 441)
    np.testing.assert_array_equal(img[1], float(stored) / 441)
    assert abs(int(stored) - exact) * 2 <= int(np.spacing(stored))
    np.testing.assert_array_equal(np.rint(img[1]), a[1])
    for z, y, x in [(0, 0, 0), (0, 16, 19), (0, 7, 9), (1, 15, 6)]:
        v = np.lib.stride_tricks.sliding_window_view(p[z, y : y + 25, x : x + 25], (5, 5))
        d = ((v - v[10, 10]) ** 2).mean((-1, -2)) / h**2
        weights = np.stack([np.exp(-d / 900), *[d <= r * r for r in (8, 16, 24, 32, 48, 64)]])
        gray = v[:, :, 2, 2]
        sums = (weights * gray).sum((1, 2))
        mass = weights.sum((1, 2))
        np.testing.assert_allclose(
            stat[z, :, y, x], np.r_[sums, mass].astype("f4"), rtol=1.5e-7, atol=1e-6
        )
        np.testing.assert_allclose(img[z, y, x], co @ sums / (co @ mass), rtol=2e-7)
    with pytest.raises(ValueError):
        blend(stat, -co)


def test_ties_retain_gaussian_and_reserves_rejected():
    for i in (4, 6):
        with pytest.raises(ValueError):
            run(i, None, None)
    c, _, _, rec = search(lambda c: dict(tp=1, fp=1, fn=1))
    np.testing.assert_array_equal(c, np.eye(7)[0])
    assert len(rec) == 48


def test_display_controls_both_required():
    rng = np.random.default_rng(95)
    a = rng.integers(256, size=(56, 77), dtype="u1")
    b = np.rint(ndi.uniform_filter(a.astype("f8"), 7)).astype("u1")
    c = ndi.median_filter(a, size=7)
    assert controls(a, b, c)[1] and not controls(a, a, c)[1] and not controls(a, b, a)[1]
    assert describe(a, a, True)["signed_error_histogram"] == {0: a.size}
