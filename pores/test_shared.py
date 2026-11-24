import numpy as np
from pore_shared_model import object_input, owned_objects


def test_whole_object_loss_uses_distinct_background_and_true_context():
    t = np.zeros((17, 32, 32), bool)
    t[4:6, 5:9, 5:9] = True
    t[4:6, 5:9, 10:14] = True
    t[0:2, 20:24, 20:24] = True
    t[10:11, 25:27, 25:27] = True
    from scipy import ndimage as ndi

    lab, _ = ndi.label(t)
    d, indices = ndi.distance_transform_edt(~t, return_indices=True)
    oracle = lab[tuple(indices)]
    records = owned_objects(t)
    assert len(records) == 2
    seen = np.zeros_like(t, dtype="u1")
    for r in records:
        j, reg, m, n = r
        np.testing.assert_array_equal(m, (oracle[reg] == j) & (d[reg] <= 2))
        seen[reg] += m
        assert n == 32 and t[reg][m].sum() == 32 and m.sum() > n
        src = np.pad(t, ((0, 0), (8, 8), (8, 8)))[:, None]
        x, take, _ = object_input(src, r)
        np.testing.assert_array_equal(x[:, 0, 8:-8, 8:-8], t[reg])
        assert np.count_nonzero(x[:, 0, 8:-8, 8:-8][take]) == 32
    assert seen.max() == 1
