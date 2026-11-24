import numpy as np
from pore_cut import apply, choose_cut
from scipy import ndimage as ndi


def brute(m, n):
    best = None
    for v in np.flatnonzero(m):
        test = m.copy()
        test.flat[v] = False
        labels, k = ndi.label(test)
        if k != 2:
            continue
        a, b = sorted(np.bincount(labels.ravel())[1:])
        if a >= n and (best is None or (a, -v) > best[:2]):
            best = a, -v, b
    if best is not None:
        return tuple(np.unravel_index(-best[1], m.shape)), best[0], best[2]


def test_low_link_matches_exhaustive_connected_graphs():
    rng = np.random.default_rng(555)
    for _ in range(50):
        lab, n = ndi.label(rng.random((3, 4, 5)) < 0.54)
        for i in range(1, n + 1):
            m = lab == i
            assert choose_cut(m, 2) == brute(m, 2)


def test_one_voxel_bridge_preserves_daughters_and_never_splits_clipped_objects():
    m = np.zeros((8, 9, 15), bool)
    m[2:6, 2:6, 2:6] = 1
    m[2:6, 2:6, 7:11] = 1
    m[4, 4, 6] = 1
    out, events = apply(m)
    assert len(events) == 1 and events[0]["daughter_voxels"] == [64, 64]
    assert np.count_nonzero(m ^ out) == 1 and not out[4, 4, 6]
    edge = m[2:]
    np.testing.assert_array_equal(apply(edge)[0], edge)
