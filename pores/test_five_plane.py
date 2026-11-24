import numpy as np
import pytest
from pore_five_plane import context


def test_true_axial_layout_and_shared_slice_control():
    rng = np.random.default_rng(53)
    v = rng.normal(size=(5, 6, 9, 11)).astype("f4")
    k = rng.normal(size=(2, 5, 6, 3, 3))
    out = context(v, "volume")
    ctrl = context(v, "slice")
    np.testing.assert_array_equal(out[:6], v[2])
    np.testing.assert_array_equal(ctrl[:6], out[:6])
    assert not ctrl[6:].any()
    assert not context(np.repeat(v[2:3], 5, axis=0), "volume")[6:].any()
    # A folded 2D kernel equals an explicit convolution on the original five
    # undifferenced planes, after absorbing the subtraction into the center.
    physical = np.empty_like(k)
    physical[:, [0, 1, 3, 4]] = k[:, 1:]
    physical[:, 2] = k[:, 0] - k[:, 1:].sum(axis=1)
    win = np.lib.stride_tricks.sliding_window_view
    planar = np.einsum("ocij,cyxij->oyx", k.reshape(2, 30, 3, 3), win(out, (3, 3), axis=(-2, -1)))
    volumetric = np.einsum("ozcij,zcyxij->oyx", physical, win(v, (3, 3), axis=(-2, -1)))
    np.testing.assert_allclose(planar, volumetric, atol=4e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        context(v[:4], "slice")
