import numpy as np
import pytest
from pore_axial_raw import PTX
from pore_reciprocal import select


@pytest.mark.skipif(
    not all((PTX / f).exists() for f in ("region-medoid.ptx", "cone-pores.exe", "pores.ptx")),
    reason="compile native tools",
)
def test_whole_boundary_handoff_symmetric_preserves_ambiguous_regions(tmp_path):
    a = np.zeros((16, 20, 38), bool)
    b = np.zeros_like(a)
    a[3:7, 3:7, 3:7] = True
    b[3:7, 3:7, 4:8] = True  # two complete 64-voxel objects, IoU = 3/5
    a[8:11, 3:6, 3:6] = b[8:11, 3:6, 3:6] = True  # shared but too small
    a[:4, 3:7, 14:18] = b[:4, 3:7, 14:18] = True  # clipped
    a[6:10, 10:14, 2:6] = True  # one parent only
    a[6:10, 10:14, 16:20] = a[6:10, 10:14, 21:25] = True
    b[6:10, 10:14, 16:25] = True  # 2:1 split, not eligible for exchange
    a[4:8, 4:8, 30:34] = True
    b[4:8, 4:8, 32:36] = True  # complete but IoU = 1/3
    outs, g = select([a, b], (1,) * 3, tmp_path, "test")
    assert g["reasons"] == {
        "replace": 1,
        "clipped": 1,
        "below 64": 1,
        "component count": 2,
        "below IoU": 1,
    }
    delta = np.zeros_like(a)
    delta[3:7, 3:7, [3, 7]] = True
    assert np.array_equal(outs[0], a ^ delta) and np.array_equal(outs[1], b ^ delta)
    assert sum(row["changed_voxels"] for row in g["regions"]) == 32
