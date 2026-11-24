import numpy as np
import pytest
from pore_section_gate import (
    choose,
    correction,
    cpu_rings,
    descriptor,
    load_source,
    match,
    planar,
    prefer,
    sections,
    solve,
)
from pore_transfer import label
from pore_xy_gpu import rings


def test_packed_cuda_rings_keep_foreign_pixels_and_separate_adjacent_planes():
    a = np.zeros((3, 13, 15), "u1")
    a[0, 3:8, 3:8] = 1
    a[0, 2:5, 8] = 2
    a[1, 7, 7] = 1
    a[2] = 2
    out = rings(a)
    for v, m in zip(out, a, strict=True):
        assert np.array_equal(v, cpu_rings(m))
    c = out[0]
    assert c[2, 8] == 7 and c[5, 5] == 0 and c[3, 3] == 2 and c[1, 3] == 6 and c[2, 3] == 5
    raw = np.where(a[0] == 1, 100, 400).astype("u2")
    p, x = descriptor(raw, c, 0.006)
    assert p == 20 and x[3:6] == [0, 1, 1] and x[-1] == 300 * 0.006 / 64


def test_clipped_parent_contributes_many_observed_sections_not_one_full_volume():
    m = np.zeros((4, 24, 24), bool)
    m[:, 6:11, 6:11] = 1
    m[:, 17:19, 19:21] = 1
    t = m.copy()
    t[:, 11, 6:11] = 1
    raw = np.where(t, 2500, 10000).astype("u2")
    rows = sections(raw, m, 0.01, truth=t)
    assert (
        len(rows) == 8
        and len({r["parent"] for r in rows}) == 2
        and all(r["clipped"] for r in rows)
    )
    assert [(r["area"], r["ref_area"]) for r in rows[:2]] == [(25, 30), (4, 4)]
    assert not label(m)[2].any()


def test_both_planar_denominators_obey_size_and_margin():
    m = np.zeros((24, 24), bool)
    m[2:5, 3:6] = 1
    m[8:10, 8:10] = 1
    m[15, 15:18] = 1
    t = m.copy()
    t[8, 8] = 0
    assert match(planar(m), planar(m)) == {2: 2} and not match(planar(m), planar(t))


def test_per_parent_balanced_ridge_and_missing_evidence():
    def row(j, y):
        return dict(x=[1] + [0] * 10, parent=j, ref_area=20 + y, area=20, perimeter=10)

    one = [row(1, 10)] * 8 + [row(2, -10)]
    two = [row(3, 5)]
    w = solve({"a": one, "b": two})
    assert w.tolist() == [250000] + [0] * 10
    assert correction(one[0], w) == 2560 and correction({}, w) == 0
    with pytest.raises(ValueError):
        solve({"a": one, "b": []})


def test_exact_half_tie_then_whole_mask_copy():
    m = np.zeros((14, 15, 16), bool)
    m[3:7, 4:8, 5:9] = 1
    m[:2, 0, 0] = 1
    e = m.copy()
    e[7, 4, 5:9] = 1
    j = str(label(m)[0][3, 4, 5])
    g = {j: {"detector": 64, "actual": 68}}
    w = np.array([1000000] + [0] * 10)
    for n, ex in [(2, m), (3, e)]:
        row = {"parent": int(j), "perimeter": n, "x": [1] + [0] * 10}
        out, log = choose(m, e, g, [row], w)
        assert np.array_equal(out, ex) and int(out.sum() - 2) == log[j]["actual"]
    assert not prefer(-2048, 68, 64) and prefer(-2049, 68, 64)


@pytest.mark.parametrize(
    "i,sl,kind",
    [
        (4, (160, 184), "segmented"),
        (6, (256, 352), "raw"),
        (2, (256, 352), "segmented"),
        (3, (400, 528), "raw"),
        (5, (608, 704), "raw"),
    ],
)
def test_source_guard_before_any_file_access(i, sl, kind):
    with pytest.raises(ValueError):
        load_source(None, i, sl, kind)
