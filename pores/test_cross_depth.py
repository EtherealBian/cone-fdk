import numpy as np
import pytest
from pore_cross_depth import eligible, median27
from scipy import ndimage as ndi


def test_uint16_order_filter_matches_scipy_including_edges_and_rail_values():
    rng = np.random.default_rng(561)
    x = rng.integers(0, 65536, (7, 13, 9), dtype="u2")
    x[:, :, 0] = 65535
    x[-3:, :, 5:] = 0
    assert np.array_equal(median27(x), ndi.median_filter(x, size=3, mode="nearest")[1:-1])
    assert np.array_equal(median27(x[1:5]), median27(x)[1:3])
    with pytest.raises(ValueError):
        median27(x.astype("i4"))


@pytest.mark.parametrize(
    "i,a,b",
    [
        (4, 608, 609),
        (6, 256, 260),
        (5, 600, 609),
        (2, 700, 705),
        (2, 400, 404),
        (3, 160, 164),
        (3, 259, 261),
    ],
)
def test_only_two_explicit_development_ranges_allowed(i, a, b):
    with pytest.raises(ValueError):
        eligible(i, a, b)
    eligible(2, 256, 260)
    eligible(5, 608, 704)
