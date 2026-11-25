import hashlib
from itertools import product

import numpy as np
import pytest
from periodic_certificate import certify
from qualify_periodic_labels import flood


def invoke(tmp, a, v, band=1, *, truth=None, mask_pin=None):
    truth = flood(a) if truth is None else truth
    m, lab_file = tmp / "mask", tmp / "labels"
    m.write_bytes(a.tobytes())
    lab_file.write_bytes(v.astype("<i4").tobytes())
    pin = hashlib.sha256(a.tobytes()).hexdigest()
    census = dict(
        observed_components=len(set(truth[truth > 0])),
        observed_flagged_cells=int((truth > 0).sum()),
        unknown_cells=int((truth == -1).sum()),
    )
    return certify(m, lab_file, mask_pin or pin, list(a.shape), census, band)


def test_all_ternary_as_stacked_independent_planes(tmp_path):
    planes = np.array(list(product(range(3), repeat=6)), dtype="u1").reshape(-1, 2, 3)
    a = np.zeros((3 * len(planes), 3), dtype="u1")
    v = np.zeros_like(a, dtype="<i4")
    for i, m in enumerate(planes):
        a[3 * i : 3 * i + 2] = m
        lab = flood(m)
        v[3 * i : 3 * i + 2] = np.where(lab > 0, lab + i * 9, lab)
    for band in (1, 2, 7, 41):
        r = invoke(tmp_path, a, v, band, truth=v)
        assert r["verified_cells"] == a.size
        assert r["labels_i32le_sha256"] == hashlib.sha256(v.tobytes()).hexdigest()


@pytest.mark.parametrize(
    "mode",
    [
        "split-seam",
        "split-band",
        "join-islands",
        "join-ends",
        "diagonal",
        "fake-root",
        "late-root",
        "zero-on-hit",
        "label-on-blind",
        "negative-other",
        "bad-mask",
        "changed-pin",
    ],
)
def test_reject_counterfeit_components(tmp_path, mode):
    a = np.array([[1, 2, 0, 0, 1], [1, 0, 1, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 0]], dtype="u1")
    truth = flood(a)
    v = truth.copy()
    if mode == "split-seam":
        v[0, 4] = 5
    elif mode == "split-band":
        v[1, 0] = 6
    elif mode == "join-islands":
        v[1, 2] = 1
    elif mode == "join-ends":
        v[3, 0] = 1
    elif mode == "diagonal":
        v[2, 3] = v[1, 2]
    elif mode == "fake-root":
        v[1, 2] = 3
    elif mode == "late-root":
        v[1, 2] = 9
    elif mode == "zero-on-hit":
        v[0, 0] = 0
    elif mode == "label-on-blind":
        v[0, 1] = 1
    elif mode == "negative-other":
        v[0, 1] = -2
    elif mode == "bad-mask":
        a[3, 0] = 3
    with pytest.raises(ValueError):
        invoke(tmp_path, a, v, truth=truth, mask_pin="0" * 64 if mode == "changed-pin" else None)
