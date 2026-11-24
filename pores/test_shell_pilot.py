import numpy as np
from pore_shell_pilot import ARMS, FACE, compare, shells, summary
from scipy import ndimage as ndi


def test_one_face_shell_only_and_inclusive_thresholds():
    p = np.zeros((15, 15, 30), "f4")
    p[3:7, 3:7, 3:7] = 0.5
    p[4, 4, 3] = 0.65
    p[4, 4, 2] = 0.35
    p[4, 4, 1] = 0.49
    p[2, 2, 2] = 0.48
    p[0:4, 8:12, 18:22] = 0.52
    p[0, 8, 17] = 0.45  # clipped: never seeds a correction
    s = shells(p)
    assert s["base"].sum() == 128
    assert np.argwhere(s["expand"] ^ s["base"]).tolist() == [[4, 4, 2]]
    assert s["contract"][4, 4, 3] and s["contract"][4, 4, 4] and not s["contract"][3, 3, 3]
    assert np.array_equal(s["contract"][:, :, 17:], s["base"][:, :, 17:])


def test_correction_can_merge_two_predictions_and_is_penalized():
    p = np.zeros((12, 12, 20), "f4")
    p[3:7, 3:7, 3:7] = 0.8
    p[3:7, 3:7, 8:12] = 0.8
    p[4, 4, 7] = 0.4
    s = shells(p)
    r = s["base"]
    report = compare(s["expand"], r, "L")
    assert ndi.label(r, FACE)[1] == 2
    assert (
        report["eligible_reference"] == 2
        and report["eligible_prediction"] == 1
        and report["matches"] == 0
    )
    assert len(report["small_references"]) == 2 and all(
        v["pred_voxels"] is None for v in report["small_references"].values()
    )


def test_fold_ids_do_not_collide_and_small_samples_cannot_pass():
    a = np.zeros((12, 12, 12), bool)
    a[3:7, 3:7, 3:7] = True
    folds = {side: {arm: compare(a, a, side) for arm in ARMS} for side in ("L", "R")}
    s = summary(folds)
    for arm in ARMS[1:]:
        assert (
            s[arm]["small_shared_ids"] == ["L:1", "R:1"]
            and s[arm]["decision"] == "insufficient pairs"
            and s[arm]["base_small_shared_p95_pct"] == 0
        )


def test_below_eligibility_is_a_real_miss_not_zero_error():
    a = np.zeros((12, 12, 12), bool)
    a[3:7, 3:7, 3:7] = True
    b = a.copy()
    b[3, 3, 3] = False
    s = compare(b, a, "L")
    assert (
        s["eligible_reference"] == 1
        and s["eligible_prediction"] == s["matches"] == 0
        and s["small_references"]["L:1"]["pred_voxels"] is None
    )


def test_whole_calibration_census_excludes_boundary_objects():
    from pore_calibration_census import census

    a = np.zeros((20, 20, 20), bool)
    a[:4, 2:6, 2:6] = True
    a[5:9, 5:9, 5:9] = True
    a[10:12, 10:14, 5:9] = True
    a[13:17, 3:7, 3:11] = True
    assert census(a) == {
        "complete_32_63": 1,
        "complete_64_127": 1,
        "complete_ge128": 1,
        "clipped_ge64": 1,
    }
