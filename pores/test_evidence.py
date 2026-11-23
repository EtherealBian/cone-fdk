import numpy as np
from review import runs
from validate import calibration, final_gates, objects, score


def test_threshold_tie_and_inclusive():
    a = np.array([1, 4, 4, 10], dtype=np.uint16)
    truth = np.array([1, 1, 1, 0], dtype=bool)
    assert calibration(a, truth) == 4
    m = score(a <= 4, truth)
    assert m["dice"] == 1 and m["porosity_error_pp"] == 0


def test_same_porosity_wrong_place_fails():
    a = np.array([1, 0], bool)
    m = score(a, ~a)
    assert m["porosity_error_pp"] == 0 and m["dice"] == 0


def test_empty_and_multirun_mask_encoding():
    assert runs(np.zeros((2, 3), int)) == []
    assert runs(np.array([[0, 4, 4], [0, 3, 2]])) == [[1, 2, 4], [4, 1, 3], [5, 1, 2]]


def test_same_centroid_and_size_not_required_for_a_match(tmp_path):
    lab = np.zeros((12, 12, 12), int)
    lab[2:6, 2:6, 2:6] = 1
    ref = np.zeros_like(lab, bool)
    ref[2:6, 2:6, 3:7] = True
    row = {"seed_z": "2", "seed_y": "2", "seed_x": "2", "id": "315", "faces": "0", "voxels": "64"}
    o = objects(lab, ref, [row], tmp_path / "matches.csv")
    m = score(lab != 0, ref)
    g = final_gates(m, o)
    assert (
        o["matches"] == 1
        and o["p95_centroid_displacement_vox"] == 1
        and o["median_abs_volume_error_pct"] == 0
    )
    assert g["porosity_absolute"] and not g["dice"] and not all(g.values())


def test_no_eligible_objects_do_not_pass(tmp_path):
    a = np.zeros((3, 3, 3), int)
    a[1, 1, 1] = 1
    row = dict(id="14", seed_x="1", seed_y="1", seed_z="1", faces="0", voxels="1")
    o = objects(a, a > 0, [row], tmp_path / "empty.csv")
    assert not all(final_gates(score(a > 0, a > 0), o).values())
