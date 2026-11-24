import numpy as np
from pore_patch_stats import ROOT, Denoise


def test_patch_statistic_cuda_against_independent_scalar_with_padding():
    rng = np.random.default_rng(123)
    a = rng.integers(0, 65000, size=(2, 17, 21), dtype=np.uint16)
    a[:, :, 6:18] = 47000
    a[0, 10:, :] = 0
    padded = np.pad(a.astype("f8"), ((0, 0), (12, 12), (12, 12)), mode="edge")
    with Denoise(ROOT / "build/pores/patch-stats.ptx") as gpu:
        for flag in (0, 1):
            actual = gpu(a, 7400, weighted=flag)
            for z, y, x in [(0, 0, 0), (1, 16, 20), (1, 5, 10), (0, 10, 7), (0, 4, 19)]:
                patches = np.lib.stride_tricks.sliding_window_view(
                    padded[z, y : y + 25, x : x + 25], (5, 5)
                )
                center = patches[10, 10]
                mean = patches.mean((-1, -2))
                var = patches.var((-1, -2))
                weight = np.exp(-np.square(patches - center).mean((-1, -2)) / 7400**2)
                if flag:
                    weight *= (
                        np.minimum(mean, mean[10, 10]) >= 0.95 * np.maximum(mean, mean[10, 10])
                    ) & (np.minimum(var, var[10, 10]) >= 0.5 * np.maximum(var, var[10, 10]))
                assert weight[10, 10] == 1
                expected = (weight * patches[:, :, 2, 2]).sum() / weight.sum()
                np.testing.assert_allclose(actual[z, y, x], expected, rtol=3e-6, atol=0.05)
