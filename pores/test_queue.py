import numpy as np
import pytest
from pore_review_queue import compare, intersect, queue
from review import runs


def test_rle_joins_recover_dense_intersections():
    rng = np.random.default_rng(449)
    a, b = rng.integers(0, 5, (2, 13, 9, 12))
    edges = intersect([runs(m) for m in a], [runs(m) for m in b])
    assert edges == {
        (i, j): np.count_nonzero((a == i) & (b == j)) for i in range(1, 5) for j in range(1, 5)
    }


def test_queue_is_not_reference_assisted_and_includes_unmatched():
    e = (
        "geometry",
        {17: 100, 38: 100, 52: 100},
        [[[10, 100, 17], [150, 100, 38], [500, 100, 52]]],
    )
    p = ("geometry", {22: 115, 47: 116}, [[[10, 115, 22], [150, 116, 47]]])
    rows = queue(e, p, p)
    assert [r[0] for r in rows] == [52, 38, 17]
    assert rows[-1][2] is False and rows[0][3] == ""
    with pytest.raises(ValueError, match="Unaligned"):
        compare(e, ("different", p[1], p[2]))
