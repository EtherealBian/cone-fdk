import numpy as np
import pytest
from pore_shell_volume import median, parents, project, select


def test_even_volume_rounds_up_and_singletons_are_unchanged():
    assert median([70, 65, 80, 69]) == 70
    assert median([64]) == 64


def test_entire_shell_update_abstains_if_removing_a_bridge():
    o = np.zeros((9, 9, 15), "u1")
    o[3:6, 3:6, 1:6] = 1
    o[3:6, 3:6, 7:12] = 1
    o[4, 4, 6] = 1
    vote = o.astype(bool)
    vote[4, 4, 6] = False
    n, reason, avail = project(o, [vote, vote], int(o.sum()) - 1, np.ones(o.shape, bool))
    assert reason == "disconnected" and avail == 1 and np.array_equal(n, o)


def test_no_growth_to_crop_face_or_close_to_other_components():
    o = np.zeros((12, 12, 12), "u1")
    o[1:5, 3:7, 3:7] = 1
    o[7, 3, 3] = 2
    vote = o == 1
    vote[0, 3, 3] = vote[5, 3, 3] = True
    ok = np.ones_like(vote)
    ok[0] = False
    n, why, k = project(o, [vote, vote], 65, ok)
    assert why == "insufficient shell" and k == 0 and np.array_equal(n, o == 1)


def test_box_with_equal_vote_ties_uses_linear_position_not_donor_order():
    o = np.zeros((9, 9, 9), "u1")
    o[2:6, 2:6, 2:6] = 1
    v = o.astype(bool)
    v[1, 2, 2] = v[6, 4, 4] = True
    n, why, k = project(o, [v, v], 65, np.ones_like(v))
    assert why == "edited" and k == 2 and n[1, 2, 2] and not n[6, 4, 4]


def test_few_or_clipped_donors_leave_detector_untouched():
    m = np.zeros((14, 14, 14), bool)
    m[2:6, 2:6, 2:6] = True
    clipped = m.copy()
    clipped[:2, 2, 2] = True
    out, g = select([m, m, clipped, clipped, clipped])
    r = g["objects"]["1"]
    assert r["reason"] == "few donors" and len(r["donors"]) == 1 and out.sum() == 64
    assert g["all_components"] == 1 and np.array_equal(m, out)


@pytest.mark.parametrize("i", [4, 6, 1])
def test_predictor_rejects_other_specimens_before_source_reads(i):
    with pytest.raises(ValueError, match="registered"):
        parents(i)
