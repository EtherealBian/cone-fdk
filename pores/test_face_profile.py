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


@pytest.mark.parametrize("i,n", [(2, 4), (3, 2), (5, 1)])
def test_complete_outside_interval_population_and_source_replay(i, n):
    from pore_axial_raw import sha
    from pore_face_profile import E
    from pore_face_profile_score import diagnostic
    from pore_region import EV, read
    from test_region_evidence import native_rows

    d = read(E / f"sample{i}.json")
    g = read(E / f"sample{i}-choices.json")
    prev = read(EV / "shell-volume" / f"sample{i}.json")
    a, b = d["arms"].values()
    h, k = [native_rows(E, i, name, v) for name, v in d["arms"].items()]
    assert set(h) == set(k)
    # In memory, native reference keys are integers, but JSON decoding returns strings.
    aa = dict(a, references={int(j): v for j, v in a["references"].items()})
    r = diagnostic(aa, g, prev)
    assert r["p95"] == d["volume_audit"]["p95"]
    assert (
        len(prev["bound"]["outside_15"]) == n
        and not d["advance"]
        and b["objects"]["p95_abs_volume_error_pct"] > 15
    )
    for rid in prev["bound"]["outside_15"]:
        _, size, ref = k[rid]
        assert 100 * abs(size - ref) <= 15 * ref and r["rows"][rid]["actual"] == size
    assert all(v["target"] == v["actual"] for v in g["objects"].values())
    repair = read(E / "repair-replay.json")["identical_fit_choices_and_masks"]
    assert len(repair) == 7 and all(sha(E / n) == v for n, v in repair.items())
    rep = read(E / "replay.json")["byte_identical_tables"]
    assert len(rep) == 12 and all(sha(E / n) == v for n, v in rep.items())
