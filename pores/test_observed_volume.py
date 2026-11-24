import numpy as np
import pytest
from pore_observed_volume import OUT, population, sha
from pore_region import read


def test_lateral_walls_and_mismatched_z_clips_not_supervised():
    m = np.zeros((12, 16, 30), bool)
    t = m.copy()
    m[:5, 1:5, 2:6] = 1
    t[1:5, 1:5, 2:6] = 1  # IoU .8 but different z contact
    m[:5, 0:4, 9:13] = 1
    t[:5, 0:4, 9:13] = 1  # lateral clip
    m[:4, 6:10, 16:20] = 1
    t[:4, 6:10, 16:20] = 1  # exactly 64 and same lower z clip
    m[1:4, 6:10, 21:25] = 1
    t[1:4, 6:10, 21:25] = 1  # too small
    d = population(m, t)
    assert d["predicted"] == d["reference"] == 2 and d["all"]["pairs"] == 1
    assert d["z_cut"]["tie"] == 1 and d["complete"]["pairs"] == 0 and not d["support"]
    assert len(d["unmatched_pred"]) == len(d["unmatched_ref"]) == 1


def test_iou_half_is_included_and_exact_15_percent_is_not_outside():
    m = np.zeros((14, 18, 16), bool)
    t = m.copy()
    m[1:5, 1:5, 1:5] = 1
    t[1:9, 1:5, 1:5] = 1
    t[2:6, 8:13, 8:13] = 1
    m |= t & (np.indices(t.shape)[1] >= 8)
    m[2, 8:11, 8:13] = False
    d = population(m, t)
    assert not d["unmatched_pred"] and not d["unmatched_ref"]
    assert [(r["residual"], r["intersection"], r["union"]) for r in d["rows"]] == [
        (64, 64, 128),
        (15, 85, 100),
    ]
    assert d["complete"] == dict(pairs=2, under=2, over=0, tie=0, outside=1)


@pytest.mark.parametrize("i", [2, 3, 5])
def test_archived_support_denominators_and_audited_tables(i):
    d = read(OUT / "support.json")["specimens"][str(i)]
    rows = d["rows"]
    assert len({r["pred"] for r in rows}) == len({r["ref"] for r in rows}) == len(rows)
    assert d["predicted"] == len(rows) + len(d["unmatched_pred"]) and d["reference"] == len(
        rows
    ) + len(d["unmatched_ref"])
    for key, group in [
        ("all", rows),
        ("complete", [r for r in rows if not r["z_faces"]]),
        ("z_cut", [r for r in rows if r["z_faces"]]),
    ]:
        assert d[key] == dict(
            pairs=len(group),
            under=sum(r["pred_voxels"] < r["ref_voxels"] for r in group),
            over=sum(r["pred_voxels"] > r["ref_voxels"] for r in group),
            tie=sum(r["residual"] == 0 for r in group),
            outside=sum(100 * abs(r["residual"]) > 15 * r["ref_voxels"] for r in group),
        )
    hashes = read(OUT / "replay.json")
    assert len(hashes) == 7 and all(sha(OUT / n) == h for n, h in hashes.items())
    assert d["support"] == (len(rows) >= 5 and min(d["all"]["under"], d["all"]["over"]) >= 2)
