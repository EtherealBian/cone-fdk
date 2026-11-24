import sys

import numpy as np
import pytest
from pore_transfer import (
    ROOT,
    boundaries,
    census,
    checked_features,
    correct,
    donor_rows,
    fit,
    label,
    pairs,
    screen,
)


def cubes():
    p = np.full((14, 14, 22), 0.1, dtype="f4")
    p[3:7, 3:7, 3:7] = 0.75
    p[3:7, 3:7, 8:12] = 0.75
    p[0:4, 8:12, 14:18] = 0.75
    p[3, 3, 2] = 0.49
    return p


def test_disjoint_faces_even_next_to_clipped_or_ineligible():
    p = cubes()
    base = p >= 0.5
    pred = label(base)
    f = boundaries(p, pred)
    assert len(f) == 2 and census(pred)["clipped_ge64"] == 1
    for inner, outer, _ in f.values():
        assert len(inner) == 56 and len(outer) == 80
        assert base.flat[inner].all() and not base.flat[outer].any()
        assert all(np.unravel_index(outer, p.shape)[2] != 7)
    p[3, 3, 1] = 0.75  # tiny foreign component disqualifies previously available outer at x=2
    assert not any(
        np.ravel_multi_index((3, 3, 2), p.shape) in o
        for _, o, _ in boundaries(p, label(p >= 0.5)).values()
    )


def test_transfer_uses_probability_order_and_deterministic_ties():
    p = cubes()
    f = boundaries(p, label(p >= 0.5))
    base = p >= 0.5
    a, rows = correct(p, f, [1.0, 0])
    assert (a ^ base).sum() == 28 and all(r["requested"] == 14 for r in rows)
    assert a[3, 3, 2]
    b, rows = correct(p, f, [-1.0, 0])
    assert (b ^ base).sum() == 28 and all(r["applied"] == -14 for r in rows)
    for inner, _, _ in f.values():
        assert (~b.flat[inner[:14]]).all() and b.flat[inner[14:]].all()
    assert not pairs(label(b), label(base), 64)  # both shrink below eligibility
    np.testing.assert_array_equal(correct(p, f, [0.0, 0])[0], base)


def test_fit_is_specimen_balanced_and_not_index_weighted():
    row = dict(voxels=100, reference_voxels=108, surface=40, u=0.2)
    other = {**row, "reference_voxels": 90, "u": -0.1}
    np.testing.assert_allclose(
        fit({2: [row] * 15, 5: [other]}), fit({2: [row], 5: [other]}), atol=1e-14
    )
    np.testing.assert_equal(fit({2: [], 5: []}), [0, 0])
    assert fit({3: [], 5: [row]})[0] > 0
    p = cubes()
    pred = label(p >= 0.5)
    rows = donor_rows(pred, pred, boundaries(p, pred))
    assert len(rows) == 2 and all(
        r["reference_voxels"] == r["voxels"] == 64 and r["iou"] == 1.0 for r in rows
    )


def test_match_eligibility_and_shared_screen_protect_denominators():
    def model(sizes, eligible_p=6, p95=20):
        return dict(
            objects=dict(
                matches=sum(v is not None for v in sizes),
                eligible_prediction=eligible_p,
                eligible_reference=len(sizes),
                p95_abs_volume_error_pct=p95,
            ),
            references={k: dict(reference_voxels=80, voxels=v) for k, v in enumerate(sizes)},
        )

    base = model([88] * 6)
    arm = model([80] * 6, p95=10)
    assert screen(base, arm, {2: 6, 3: 5})["decision"] == "pass"
    assert screen(base, arm, {2: 6, 3: 4})["decision"] == "insufficient support"
    assert screen(base, model([80] * 5 + [None], 5, 10), {2: 6, 3: 5})["decision"] == "fail"
    assert not screen(base, model([80] * 6, 6, 21), {2: 6, 3: 5})["criteria"][
        "full_p95_not_worse"
    ]


@pytest.mark.skipif(
    sys.platform != "win32" or not (ROOT / "build/pores/transfer_faces.ptx").exists(),
    reason="built Windows CUDA kernel required",
)
def test_compiled_cuda_face_ownership_matches_independent_scipy():
    p = cubes()
    p[3, 3, 1] = 0.75
    assert len(checked_features(p, label(p >= 0.5))) == 2


def test_review_encoding_preserves_every_bit_and_raw_gray():
    import base64
    import io

    from PIL import Image
    from pore_transfer_view import pack_plane

    gray = np.arange(8, dtype="u1").reshape(2, 4) * 31
    f = np.arange(8, dtype="u1").reshape(2, 4)
    layers = [(f & k) != 0 for k in (1, 2, 4)]
    with Image.open(io.BytesIO(base64.b64decode(pack_plane(gray, *layers)))) as image:
        rgb = np.asarray(image)
        np.testing.assert_array_equal(rgb[:, :, 0], gray)
        np.testing.assert_array_equal(rgb[:, :, 1], f << 5)


def test_exact_rounding_tie_is_even_not_always_up():
    p = np.array([[[0.7, 0.7, 0.7, 0.7, 0.2, 0.3]]], dtype="f4")
    ids = np.array([0, 1, 2, 3])
    out = np.array([4, 5])
    f = {1: (ids, out, 0.0)}
    # 4*0.125 = half a voxel: keep the original mask; 4*0.25 = 1: prefer 0.3.
    a, c = correct(p, f, [0.125, 0])
    assert c[0]["applied"] == 0
    b, c = correct(p, f, [0.25, 0])
    assert c[0]["applied"] == 1 and b.ravel().tolist() == [True, True, True, True, False, True]
