import copy
from itertools import product

import numpy as np
import pytest
from bridge_design import protocol
from bridge_phantom import chords
from validate_bridge import assess, census


@pytest.mark.parametrize(
    "z,depth,cut",
    [
        (-3, 0.18, 0),
        (-2.9, 0.18, 0.72),
        (-0.7, 0.18, 0.18),
        (0.7, 0.18, 0.72),
        (2.9, 0.18, 0),
        (0, None, 0),
        (0, 0.30, 0.30),
    ],
)
def test_axial_parallel_ray_and_shared_slab_boundaries(z, depth, cut):
    spec = protocol()
    phi = np.deg2rad(46)
    u = np.array([np.cos(phi), np.sin(phi), 0])
    src = -20 * u + [0, 0, z]
    assert chords(src, u, spec, depth) == pytest.approx(2 * (9.28 - 5.92) - cut, abs=1e-10)


@pytest.mark.parametrize(
    "s,d",
    [
        ((-16, -12, -2.6), (1, 0.8, 0.12)),
        ((-16, -12, -2.6), (1, 0.8, 0.2)),
        ((18, 11, 3.7), (-0.95, -0.67, -0.07)),
        ((22, 1, -1), (-0.9, 0.1, 0.16)),
        ((18, 0, 3.0), (0, 1, -0.1)),
    ],
)
def test_chord_with_independent_cartesian_midpoint_integral(s, d):
    t = (np.arange(150000) + 0.5) * 0.0003
    xyz = np.array(s)[:, None] + np.array(d)[:, None] * t
    x, y, z = xyz
    rr = np.hypot(x, y)
    phi = np.deg2rad(46)
    depth = np.select(
        [(-2.9 <= z) & (z < -0.7), (-0.7 <= z) & (z < 0.7), (0.7 <= z) & (z < 2.9)],
        [0.72, 0.3, 0.72],
        default=0.0,
    )
    inside = (
        (rr <= 9.28)
        & (rr >= 5.92)
        & (abs(z) <= 7)
        & (x * np.cos(phi) + y * np.sin(phi) <= 9.28 - depth)
    )
    assert chords(np.array(s), np.array(d), protocol(), 0.3) == pytest.approx(
        inside.sum() * 0.0003, abs=0.0012
    )


def test_bad_cut_rejected():
    with pytest.raises(ValueError, match="bore"):
        chords(np.array([0, 20, 0]), np.array([0, 1, 0]), protocol(), 3.5)


def test_separate_island_does_not_determine_anchor_connectivity():
    labels = np.array([[1, 0, 1], [0, 4, 0]], dtype="i4")
    anchors = [[0, 0], [0, 2]]
    assert census(labels, labels > 0, anchors) == dict(
        anchor_ids=[1, 1],
        anchors_both_flagged=True,
        unknown_cells=0,
        connected=True,
        footprint_iou=1,
        observed_flagged_cells=3,
        components=2,
    )
    labels[0, 2] = 3
    with pytest.raises(ValueError, match="reachability"):
        census(labels, labels > 0, anchors)


def test_indeterminate_does_not_become_disconnected():
    for labels in (np.array([[1, -2, 1]]), np.array([[1, 0, 0]])):
        assert census(labels, labels > 0, [[0, 0], [0, 2]])["connected"] is None


def sample():
    cases = {}
    for name in ("open-control", "joined-control", "critical-a", "critical-b"):
        rows = [
            dict(
                pair=[i, j],
                anchors_both_flagged=True,
                unknown_cells=0,
                footprint_iou=0.90,
                peak_error_mm=0.05,
                connected=name == "joined-control" or (name == "critical-a" and i == j),
            )
            for i, j in product(range(3), repeat=2)
        ]
        cases[name] = dict(nominal_pair=[1, 1], rows=rows)
    return cases


def test_within_acquisition_not_between_seeds():
    cases = sample()
    assert assess(cases, protocol())["passed"]
    for row in cases["critical-a"]["rows"]:
        row["connected"] = True
    assert not assess(cases, protocol())["gates"]["critical-within-acquisition-switch"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("peak_error_mm", -0.07),
        ("footprint_iou", 0.84999),
        ("connected", True),
        ("unknown_cells", 1),
        ("anchors_both_flagged", False),
    ],
)
def test_scientific_failure_is_returned_as_data(key, value):
    cases = sample()
    cases["open-control"]["rows"][4][key] = value
    assert not assess(cases, protocol())["passed"]


def test_duplicate_pair_not_qualified():
    cases = sample()
    rows = cases["critical-b"]["rows"]
    rows[8] = copy.deepcopy(rows[7])
    with pytest.raises(RuntimeError, match="inventory"):
        assess(cases, protocol())
