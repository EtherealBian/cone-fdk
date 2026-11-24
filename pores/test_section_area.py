import numpy as np
from pore_section_area import draw, sections
from scipy import ndimage as ndi


def test_sections_keep_clipped_parent_but_exclude_clipped_planar_components():
    t = np.zeros((8, 24, 24), bool)
    t[:, 4:7, 4:7] = 1
    t[0, 4:7, 10:13] = 1
    t[0, 5, 7:10] = 1
    t[1:6, 4:7, 10:13] = 1
    t[3, 15:17, :4] = 1
    t[5, 15:17, 13:15] = 1
    t[4, 21, 15] = 1
    g, info = sections(t)
    assert len(g) == 2
    parent = next(k for k, r in info.items() if r["touches_z"])
    assert info[parent]["touches_face"]
    assert sum(r[0] == 1 for r in g[int(parent)]) == 2
    seen = np.zeros(t.shape, "u1")
    for rows in g.values():
        for z, j, y, u, mask, n in rows:
            lab, _ = ndi.label(t[z])
            coords = np.argwhere(lab == j)
            other = np.argwhere((lab != 0) & (lab != j))
            assert n == len(coords)
            for a, b in np.ndindex(mask.shape):
                at = np.array([a + y.start, b + u.start])
                dist = ((coords - at) ** 2).sum(1).min()
                competitor = ((other - at) ** 2).sum(1).min() if len(other) else np.inf
                if dist != competitor:
                    assert mask[a, b] == (dist <= 4 and dist < competitor)
            seen[z, y, u] += mask
    assert seen.max() == 1 and not np.any(seen[3, 15:17, :4])
    assert len(g[int(parent)]) == 13
    a, b = np.random.default_rng(1), np.random.default_rng(1)
    for _ in range(20):
        ka, ra = draw(g, a)
        kb, rb = draw(g, b)
        assert (ka, ra[:2]) == (kb, rb[:2])


def test_screen_includes_coverage_and_common_id_support():
    import copy

    import pytest
    from pore_section_score import compare

    ref = {k: {"reference_voxels": 100, "voxels": 110 + k} for k in range(5)}
    b = {
        "references": ref,
        "metrics": {"dice": 0.98},
        "objects": {"matched_fraction_reference": 0.9, "matched_fraction_prediction": 0.95},
    }
    a = copy.deepcopy(b)
    for row in a["references"].values():
        row["voxels"] = 100
    assert compare(b, a)["advance"]
    a["objects"]["matched_fraction_reference"] = 0.89
    assert not compare(b, a)["advance"]
    a["references"][1]["voxels"] = None
    assert not compare(b, a)["criteria"]["support"]
    a["references"][1]["reference_voxels"] = 70
    with pytest.raises(ValueError):
        compare(b, a)
