import numpy as np
import pytest
from pore_raw_grid import GridGPU, cpu_grid, expand, scalar


@pytest.mark.parametrize("s", [1, 4])
def test_scanner_phase_halo_and_independent_scalar(s):
    rng = np.random.default_rng(493)
    raw = rng.integers(0, 65536, (3, 92, 112), dtype="u2")
    full = expand(cpu_grid(raw, s, 3), s)
    for y, x in ((0, 0), (1, 7), (12, 37), (59, 79)):
        np.testing.assert_allclose(full[:, y, x], scalar(raw, s, 3, y, x), atol=1e-5, rtol=0)
    for y, x in ((0, 0), (4, 12), (28, 48)):
        tile = expand(cpu_grid(raw[:, y : y + 64, x : x + 64], s, 3), s)
        np.testing.assert_array_equal(tile, full[:, y : y + 32, x : x + 32])
    # An arbitrary tile offset in the output plane must not change the sampling lattice.
    g = cpu_grid(raw, s, 3)
    np.testing.assert_array_equal(expand(g[:, 3:51, 1:49], s), full[:, 3:35, 1:33])
    np.testing.assert_array_equal(expand(cpu_grid(raw[::-1], s, 3), s), full)


@pytest.mark.parametrize("s", [1, 4])
def test_two_scale_cuda_box_fields(s):
    from pore_axial_raw import PTX

    if not (PTX / "raw-grid.ptx").exists():
        pytest.skip("Requires compiled CUDA grid")
    raw = np.random.default_rng(91).integers(0, 65536, (3, 52, 76), dtype="u2")
    with GridGPU() as gpu:
        actual = gpu(raw, s, 5)
    np.testing.assert_allclose(actual, cpu_grid(raw, s, 5), rtol=0, atol=1e-6)
