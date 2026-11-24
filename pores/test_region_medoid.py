
import numpy as np
import pytest
from pore_axial_raw import PTX
from pore_region import consensus, scores
from scipy import ndimage as ndi


def test_exact_jaccard_accounts_for_empty_pairs():
    sz = np.array([1, 1, 1, 0, 0, 0, 0])
    inter = np.diag(sz)
    s = scores(inter, sz)
    assert s == [1, 1, 1, 4, 4, 4, 4]


@pytest.mark.skipif(
    not (PTX / "region-medoid.ptx").is_file(), reason="Compile optional CUDA region study"
)
def test_union_bridge_tie_order_multiple_regions_and_empty():
    m = np.zeros((7, 3, 5, 12), bool)
    for j in range(7):
        m[j, 1, 2, 2:9] = True
        m[j, 1, 2, 2 + j] = False
    m[6, 0, 4, 0] = True
    lab, _ = ndi.label(m.any(0))
    major, med, regions = consensus(m, lab)
    assert major[1, 2, 2:9].all() and not any(np.array_equal(major, a) for a in m)
    assert np.array_equal(med[1, 2], m[0, 1, 2]) and not med[0, 4, 0] and len(regions) == 2
    assert regions[1]["tied"] == list(range(7)) and regions[1]["area"] == 7
    perm = [3, 0, 1, 2, 4, 5, 6]
    _, alt, _ = consensus(m[perm], lab)
    assert np.array_equal(alt[1, 2], m[3, 1, 2])
    m[:] = False
    empty = consensus(m, np.zeros_like(lab))
    assert not empty[0].any() and not empty[1].any() and not empty[2]
