import numpy as np
import pytest
from pore_face_profile import choices, faces, features, raw_slab, select


def test_profile_reversals_axis_permutation_and_contrast_normalization():
    a = np.random.default_rng(811).integers(100, 900, size=(18, 21, 23), dtype="u2")
    p = np.array([[1, 2, 3], [5, 8, 10], [0, 0, 0]])
    v = features(a, p, 2, -1)
    assert np.array_equal(v, features(a * 3 + 59, p, 2, -1))
    assert np.array_equal(v, features(a.transpose(2, 1, 0), p[:, ::-1], 0, -1))
    q = p.copy()
    q[:, 2] = a.shape[2] - 11 - q[:, 2]
    assert np.array_equal(v, features(a[:, :, ::-1], q, 2, 1))


def test_constant_and_single_flat_channel_use_zero_then_stay():
    a = np.full((20, 20, 20), 65000, "u2")
    p = np.array([[4, 3, 5]])
    v = features(a, p, 0, 1)
    assert not v.any()
    d, g = choices(v, np.ones(16, "i8"))
    assert d.tolist() == [0] and not g.any()
    v[:, (0, 2), 0] = 1
    d, g = choices(v, np.ones(16, "i8"))
    assert d.tolist() == [-1] and g.tolist() == [1]


def test_internal_faces_only_and_flat_source_does_not_move_pore():
    m = np.zeros((12, 13, 14), bool)
    m[3:7, 4:8, 5:9] = True
    m[0, 1, 1] = True
    assert sum(len(p) for _, _, p in faces(m)) == 96 + 5
    a = np.zeros(tuple(n + 10 for n in m.shape), "u2")
    v, g = select(a, m, np.ones(16, "i8"))
    assert (
        np.array_equal(v, m)
        and g["oriented_faces"] == 96
        and g["components"] == 2
        and g["reasons"] == {"equal": 1}
    )


@pytest.mark.parametrize(
    "i,sl", [(4, (160, 184)), (6, (256, 352)), (2, (608, 704)), (3, (400, 528))]
)
def test_raw_guard_precedes_all_archive_access(i, sl):
    with pytest.raises(ValueError, match="Unregistered"):
        raw_slab(None, i, sl)
