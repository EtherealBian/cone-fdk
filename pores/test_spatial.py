import numpy as np
from pore_spatial_gpu import forward
from pore_spatial_model import ROOT, batch, pools


def test_cuda_dilated_valid_convolution_against_direct_cpu():
    rng = np.random.default_rng(105)
    x = rng.normal(size=(2, 3, 14, 15)).astype("f4")
    w = rng.normal(size=(2, 3, 3, 3)).astype("f4")
    b = np.array([1, -2], "f4")
    w1 = np.array([[[[0.4]], [[0.6]]]], "f4")
    b1 = np.array([-0.3], "f4")
    y = forward(x, [(w, b, 2), (w1, b1, 1)], ROOT / "build/pores/spatial.ptx")
    blocks = np.lib.stride_tricks.sliding_window_view(x, (5, 5), axis=(2, 3))[
        :, :, :, :, ::2, ::2
    ]
    mid = np.maximum(0, np.einsum("ncyxij,ocij->noyx", blocks, w) + b[None, :, None, None])
    expected = 0.4 * mid[:, 0] + 0.6 * mid[:, 1] - 0.3
    np.testing.assert_allclose(y[:, 0], expected, rtol=2e-5, atol=5e-6)


def test_training_tiles_are_aligned_after_every_augmentation():
    z, y, x = np.indices((4, 256, 256))
    truth = (z == 2) & ((x - 166) ** 2 + (y - 80) ** 2 < 35)
    maps = np.pad(truth[:, None].astype("f4"), [(0, 0), (0, 0), (8, 8), (8, 8)])
    sets = pools(truth)
    rng = np.random.default_rng(12)
    for i in range(3):
        xx, t = batch(maps, truth, rng, sets, i)
        np.testing.assert_array_equal(xx[:, :, 8:-8, 8:-8], t)
