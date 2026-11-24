import numpy as np
import pytest
from pore_slice_noise import bands, measures, scales


def test_probe_order_statistic_replay_and_zero_noise_guard():
    a = np.full((3, 34, 34), 40000, dtype="u2")
    assert measures(a) == 0
    a[1, 1:-1:2, 1:-1] = 0
    assert measures(a) == 20000
    fit = np.full((8, 4), 2.0)
    ck = fit.copy()
    ck[0] = 0.25
    ck[1] = 48
    ck[2] = 0.1, 2, 2, 56
    np.testing.assert_array_equal(bands(fit, ck), [15, 60, 30, 30, 30, 30, 30, 30])
    assert bands(0 * fit, ck) is None
    for i in (4, 6):
        with pytest.raises(ValueError):
            scales(None, i, 160)
