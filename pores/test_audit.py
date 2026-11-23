import numpy as np
import pytest
from audit import decode


def test_decode_refuses_overlaps_binary_wrong_ids_and_bad_lengths():
    for malformed in (
        [[0, 2, 1], [1, 1, 1]],
        [[0, 1, 2]],
        [[3, 2, 1]],
        [[0, -1, 1]],
        [[0, 1, 1.5]],
    ):
        with pytest.raises(ValueError):
            decode(malformed, 4, True)
    np.testing.assert_equal(decode([[1, 2, 7]], 4), [0, 7, 7, 0])


def test_case_points_are_real_disagreements_on_worst_slice():
    from cases import eligible, peak

    a = np.zeros((6, 9, 12), bool)
    a[1, 3, 4] = 1
    a[2, 4, 2:7] = 1
    z, y, x = peak(a, a)
    assert z == 2 and a[z, y, x]
    z, y, x = peak(np.zeros_like(a), a)
    assert a[z, y, x]
    lab = np.zeros((10, 11, 12), "i4")
    lab[2:6, 3:7, 2:6] = 4
    lab[1, 1, 1] = 7
    lab[:2, 6:, 2:] = 9
    assert eligible(lab) == {4}


def test_gate_never_rounds_up_a_near_miss():
    from validate import final_gates

    o = dict(
        matched_fraction_prediction=0.9,
        matched_fraction_reference=0.9,
        median_abs_volume_error_pct=5.0,
        p95_abs_volume_error_pct=15.0,
        median_abs_diameter_error_pct=2.0,
        p95_centroid_displacement_vox=1.0,
    )
    m = dict(
        dice=0.95, precision=0.93, recall=0.93, reference_porosity_pct=1.0, porosity_error_pp=0.05
    )
    assert all(final_gates(m, o).values())
    assert not final_gates(dict(m, dice=np.nextafter(0.95, 0)), o)["dice"]
    assert not final_gates(dict(m, reference_porosity_pct=0.1), o)["porosity_relative"]
    assert not final_gates(m, dict(o, p95_abs_volume_error_pct=np.nextafter(15.0, 16.0)))[
        "p95_abs_volume_error_pct"
    ]
