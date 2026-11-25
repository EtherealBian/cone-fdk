import numpy as np
import pytest
from bridge_design import linked, loss_field, predict, protocol


def test_preregistered_outcomes_and_critical_equality():
    d = predict()["cases"]
    assert {
        k: (v["flagged_cells"], v["central_row_flagged_cols"], v["anchors_connected"])
        for k, v in d.items()
    } == {
        "baseline": (0, 0, False),
        "open-control": (770, 0, False),
        "critical-a": (777, 1, True),
        "critical-b": (777, 1, True),
        "joined-control": (903, 19, True),
    }
    spec = protocol()
    f = loss_field(spec, 0.30)
    assert f[20, 46] == 0.30 and f[20, 45] < 0.30 and f[20, 47] < 0.30
    ts = spec["scans"]
    assert len({s["seed"] for s in ts}) == 5


@pytest.mark.parametrize("depth", [0.18, 0.30, 0.42])
def test_direct_cartesian_halfspace(depth):
    spec = protocol()
    f = loss_field(spec, depth)
    R = spec["tube_mm"]["outer_radius"]
    normal = np.array([np.cos(np.deg2rad(46)), np.sin(np.deg2rad(46))])
    radii = np.linspace(R - 0.75, R, 1501)
    for y, x in [(20, 46), (20, 43), (0, 46), (11, 52), (29, 45), (29, 225), (20, 359)]:
        z = (y - 20) * 0.2
        local = 0 if abs(z) > 2.9 else 0.72 if abs(z) > 0.7 else depth
        points = radii[:, None] * np.array([np.cos(np.deg2rad(x)), np.sin(np.deg2rad(x))])
        removed = points @ normal > R - local
        assert abs(removed.sum() * 0.0005 - f[y, x]) <= 0.000501


def test_connectivity_has_one_periodic_axis_and_no_diagonal_step():
    mask = np.zeros((3, 5), bool)
    mask[0, [0, -1]] = True
    mask[2, 0] = True
    mask[1, 1] = True
    assert (
        linked(mask, [[0, 4], [0, 0]])
        and not linked(mask, [[0, 0], [2, 0]])
        and not linked(mask, [[0, 0], [1, 1]])
    )
    with pytest.raises(ValueError, match="Boolean"):
        linked(mask.astype("i4"), [[0, 0], [2, 0]])
    with pytest.raises(ValueError, match="outside"):
        linked(mask, [[-1, 0], [2, 0]])
