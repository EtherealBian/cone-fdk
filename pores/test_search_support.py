import numpy as np
import pytest
from pore_search_support import ROOT, Denoise, run


def test_only_development():
    for i in (4, 6):
        with pytest.raises(ValueError):
            run(i, None, None)


def test_search_support_native_vs_double_border_and_geometry():
    a = np.random.default_rng(918).integers(0, 65000, (2, 16, 19), dtype="u2")
    a[1, :, :] = 47512
    pad = np.pad(a.astype("f8"), ((0, 0), (12, 12), (12, 12)), mode="edge")
    j, i = np.ogrid[-10:11, -10:11]
    disk = i * i + j * j <= 100
    assert disk.sum() == 317
    with Denoise(ROOT / "build/pores/search-support.ptx") as gpu:
        for flag in (0, 1):
            v = gpu(a, 9000, weighted=flag)
            np.testing.assert_array_equal(v[1], a[1])
            for y, x in [(0, 0), (15, 18), (6, 9), (0, 17)]:
                patches = np.lib.stride_tricks.sliding_window_view(
                    pad[0, y : y + 25, x : x + 25], (5, 5)
                )
                delta = patches - patches[10, 10]
                w = np.exp(-np.square(delta).mean((-1, -2)) / 9000**2)
                if flag:
                    w *= disk
                expected = (w * patches[:, :, 2, 2]).sum() / w.sum()
                np.testing.assert_allclose(v[0, y, x], expected, rtol=3e-6, atol=0.05)
