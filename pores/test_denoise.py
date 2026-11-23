import numpy as np
from denoise import Denoise
from pore_check import ROOT


def test_cuda_patch_filter_matches_cpu_at_edges():
    a = np.array(
        [[[3.0, 7.0, 8.0, 1.0], [4.0, 2.0, 11.0, -1.0], [0.0, 0.5, 4.0, 6.0]]], dtype="f4"
    )
    h = 3.5
    expected = np.empty_like(a)
    p = np.pad(a[0], 12, mode="edge")
    for y, x in np.ndindex(a.shape[1:]):
        Y, X = y + 12, x + 12
        target = p[Y - 2 : Y + 3, X - 2 : X + 3]
        weights = []
        vals = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                neighbor = p[Y + j - 2 : Y + j + 3, X + i - 2 : X + i + 3]
                weights.append(np.exp(-np.mean((target - neighbor) ** 2) / h**2))
                vals.append(p[Y + j, X + i])
        expected[0, y, x] = np.average(vals, weights=weights)
    with Denoise(ROOT / "build/pores/nlm.ptx") as g:
        result = g(a, h)
        np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)


def test_patch_offsets_estimate_target_cpu_and_gpu():
    from block_model import aggregate

    a = np.arange(2 * 6 * 7, dtype="f4").reshape(2, 6, 7) / 7
    # Each input patch centred at (y,x) describes (y+dy,x+dx). Only the interior is promised.
    field = np.stack(
        [np.roll(a, (-dy, -dx), (1, 2)) for dy in range(-2, 3) for dx in range(-2, 3)], axis=1
    )
    np.testing.assert_allclose(aggregate(field, 2)[:, 2:-2, 2:-2], a[:, 2:-2, 2:-2], rtol=1e-6)
    with (
        Denoise(ROOT / "build/pores/patches.ptx", patch=True) as patch,
        Denoise(ROOT / "build/pores/nlm.ptx") as pixel,
    ):
        np.testing.assert_allclose(patch(a, 1)[:, 12], pixel(a, 1), rtol=5e-6, atol=2e-5)


def test_multiscale_center_channel_reproduces_single_scale_cuda():
    a = np.random.default_rng(44).normal(size=(3, 11, 19)).astype("f4")
    with (
        Denoise(ROOT / "build/pores/multiscale.ptx", multi=True) as multi,
        Denoise(ROOT / "build/pores/nlm.ptx") as single,
    ):
        result = multi(a, 1.0)
        assert result.shape == (3, 6, 11, 19)
        for k, bw in enumerate([0.7, 1.0, 1.3]):
            np.testing.assert_allclose(result[:, k], single(a, bw), rtol=2e-5, atol=2e-5)
