import numpy as np
import pytest
from region_review import labels
from stripe_frontier import Frontier


def reference(mask, pts):
    a = labels(mask)
    ids, n = np.unique(a[a > 0], return_counts=True)
    x, y = (a[tuple(t)] for t in pts)
    return dict(
        components=len(ids),
        largest_cells=int(n.max(initial=0)),
        flagged_cells=int(mask.sum()),
        anchors_linked=bool(x and x == y),
    )


def compare(mask, rows, pts):
    f = Frontier(mask.shape[1], pts)
    for i in range(0, len(mask), rows):
        f.feed(mask[i : i + rows])
        assert len(f.flags) == len(f.size) <= mask.shape[1] + 1
    assert f.result() == reference(mask, pts)


@pytest.mark.parametrize("rows", [1, 2, 5, 13, 90])
def test_band_partition_does_not_change_cylindrical_components(rows):
    rng = np.random.default_rng(17631)
    for occupancy in (0, 0.05, 0.34, 0.5, 0.94, 1):
        for h, w in ((1, 21), (18, 19), (15, 101)):
            mask = rng.random((h, w)) < occupancy
            compare(mask, rows, [[0, 0], [h - 1, w - 1]])


def test_late_union_joins_earlier_fragments_and_finalized_anchors():
    mask = np.zeros((13, 17), bool)
    mask[1:10, [1, 5, 9]] = True
    mask[9, 1:10] = True
    mask[6, [0, 16]] = True
    mask[2:7, 16] = True
    mask[10:12, 11] = True
    for pts in ([[2, 1], [2, 9]], [[2, 16], [2, 9]], [[2, 1], [11, 11]], [[12, 0], [2, 9]]):
        for rows in range(1, 14):
            compare(mask, rows, pts)


def test_checkerboard_and_open_axial_ends():
    mask = np.indices((34, 128)).sum(0) % 2 == 0
    compare(mask, 1, [[0, 0], [33, 127]])
    mask[:] = False
    mask[[0, 33], :] = True
    compare(mask, 1, [[0, 0], [33, 0]])


def test_band_or_anchor_error_not_silently_coerced():
    with pytest.raises(ValueError):
        Frontier(12, [[0, 0], [0, 0]])
    f = Frontier(12, [[0, 0], [4, 0]])
    with pytest.raises(ValueError):
        f.feed(np.ones((3, 12), "i4"))
    with pytest.raises(ValueError):
        f.feed(np.zeros((0, 12), bool))
    f.feed(np.zeros((3, 12), bool))
    with pytest.raises(ValueError):
        f.result()
