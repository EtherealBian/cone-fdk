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


@pytest.mark.parametrize(
    "i,outside,changed", [(2, [22, 56, 63, 87], 447), (3, [71, 137], 103), (5, [26], 18)]
)
def test_sealed_projection_and_attainable_interval(i, outside, changed):
    from fractions import Fraction

    from pore_axial_raw import digest, sha
    from pore_axial_score import read_mask
    from pore_region import read
    from pore_shell_volume import E
    from pore_shell_volume_score import bounds
    from test_region_evidence import native_rows

    s = read(E / f"sample{i}.json")
    g = read(E / f"sample{i}-choices.json")
    b = s["bound"]
    p = read(E / "predictions.json")["specimens"][str(i)]
    assert (
        b == bounds(s["arms"]["curvature"], g)
        and b["outside_15"] == outside
        and b["p95"]["interval"] > 15
    )
    assert not s["comparison"]["lost"] and not s["comparison"]["gained"] and not s["advance"]
    assert sum(r["changed"] for r in g["objects"].values()) == changed
    assert all(
        b["rows"][str(k)]["reference"] > max(b["rows"][str(k)]["interval"])
        and Fraction(b["rows"][str(k)]["percent_errors"]["interval"]) > 15
        for k in outside
    )
    hits = [native_rows(E, i, k, a) for k, a in s["arms"].items()]
    assert set(hits[0]) == set(hits[1])
    assert (
        digest(read_mask(E, i, "shell", p["masks"]["shell"]))
        == s["arms"]["shell"]["mask_u8_sha256"]
    )
    for f, n in [("prediction-replay.json", 7), ("replay.json", 12)]:
        rec = read(E / f)
        d = rec if n == 7 else rec["byte_identical_tables"]
        assert len(d) == n and all(sha(E / k) == h for k, h in d.items())
