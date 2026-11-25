"""Material frame registration: proper handedness, observability, gating and scanner axes."""

import json

import numpy as np
import pytest
from common import ROOT, Geometry
from remount_datum import (
    DatumRejected,
    candidates,
    panel_encloses,
    read_calibration,
    register,
    scout_slab,
)
from remount_loss import run, scanner_grid
from scipy.spatial.transform import Rotation
from validate_cylinder import spec

CFG = json.loads((ROOT / "fixtures/remount-fixture.json").read_bytes())
X = np.array(CFG["beads_material_xyz_mm"])
G = Geometry(120, 128, 480, 0.14, 0.24, 120, 205)
R = Rotation.from_rotvec([1.7, -0.38, 0.53]).as_matrix()
T = np.array([-0.37, 1.8, 0.43])
FIT = dict(rotation=R, translation_mm=T)


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [1, 3, 2, 0], [3, 2, 0, 1]])
def test_unique_proper_correspondence(order):
    obs = (X @ R.T + T)[order]
    f = register(X, obs, 0.035)
    np.testing.assert_array_equal(np.asarray(order)[f["candidate_for_template"]], np.arange(4))
    np.testing.assert_allclose(f["rotation"], R, atol=1e-14)
    np.testing.assert_allclose(f["translation_mm"], T, atol=1e-14)
    assert f["max_residual_mm"] < 1e-13 and f["next_permutation_max_residual_mm"] > 0.07


@pytest.mark.parametrize(
    "case,reason",
    [
        ("mirror", "fixture_residual"),
        ("shift_one", "fixture_residual"),
        ("symmetry", "ambiguous_fixture"),
        ("nearly_symmetric", "ambiguous_fixture"),
        ("plane", "degenerate_fixture"),
        ("line", "degenerate_fixture"),
    ],
)
def test_reject_nonidentifiable(case, reason):
    a = X.copy()
    b = a.copy()
    if case == "mirror":
        b[:, 0] *= -1
    if case == "shift_one":
        b[0, 1] += 0.3
    if "symmet" in case:
        a = np.asarray([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * 4
        if case == "nearly_symmetric":
            a[0, 0] += 0.045
        b = a.copy()
    if case == "plane":
        a[:, 2] = 0
    if case == "line":
        a[:, 1:] = 0
    with pytest.raises(DatumRejected, match=reason):
        register(a, b, 0.035)


@pytest.mark.parametrize(
    "key,val",
    [
        ("locator_n", True),
        ("locator_n", 4096),
        ("bead_radius_mm", 0.1),
        ("fit_limit_mm", float("nan")),
        ("bead_level_per_mm", 0),
        ("beads_material_xyz_mm", [[1, 0, 0]] * 4),
    ],
)
def test_bad_calibration(key, val):
    with pytest.raises(ValueError):
        read_calibration({**CFG, key: val})


def scout(offset=0):
    n = 64
    vol = np.zeros((n, n, n), "<f4")
    ijk = np.indices(vol.shape)
    pts = np.array([[20, 10, 24], [46, 42, 16], [18, 43, 40], [43, 20, 36]])
    for i, p in enumerate(pts):
        if i == 0:
            p[0] += offset
        vol[((ijk - p[:, None, None, None]) ** 2).sum(axis=0) <= 3**2] = 0.24
    return vol, pts


def test_parent_frame_after_scout_crop():
    vol, pts = scout()
    args = (12.8, 0.12, 0.6)
    full = candidates(vol, *args)
    small = candidates(vol[10:-10], *args, z0=10)
    assert full == small
    xyz = np.asarray([b["scanner_xyz_mm"] for b in full])
    np.testing.assert_allclose(xyz, (pts[np.argsort(pts[:, 0]), ::-1] - 31.5) * 0.2, atol=1e-13)


@pytest.mark.parametrize(
    "case,reason",
    [
        ("missing", "candidate_count:3"),
        ("spare", "candidate_count:5"),
        ("face", "scout_edge"),
        ("nonfinite", "invalid_scout"),
        ("shape", "non_spherical"),
    ],
)
def test_bad_scout(case, reason):
    vol, pts = scout(offset=-7 if case == "face" else 0)
    if case == "missing":
        vol[:25, :20, :32] = 0
    if case == "spare":
        vol[30:35, 30:35, 30:35] = 0.24
    if case == "shape":
        vol[:25, :20, :32] = 0
        vol[17:21, 4:16, 22:25] = 0.24
    if case == "nonfinite":
        vol[22, 22, 22] = np.nan
    with pytest.raises(DatumRejected, match=reason):
        candidates(vol[10:-10], 12.8, 0.12, 0.6, z0=10)


def test_visibility_and_bounds():
    assert panel_encloses(G, np.array([0.3, 0.2, 0.1]), lambda v: 0.65)
    assert not panel_encloses(G, np.array([0.3, 0.2, 12.0]), lambda v: 0.65)
    assert not panel_encloses(G, np.array([24.0, 0, 0]), lambda v: 0.65)
    lo, depth = scout_slab(160, 32, 0.65, G)
    assert 0 < lo < 80 and depth + 2 * lo == 160


def test_grid_rotates_vectors_and_translates_only_origin(tmp_path):
    g = spec(tmp_path / "a", (36, 9, 20), (0.1, -0.3, 0.6), (1, 0, 0), (0, 1, 0), (0.1, 0.1, 0.2))
    out = tmp_path / "b"
    scanner_grid(g, out, FIT, G)
    center, axis, seam, steps = np.asarray(out.read_text().split()[4:], dtype=float).reshape(4, 3)
    np.testing.assert_allclose(center, R @ [0.1, -0.3, 0.6] + T)
    np.testing.assert_allclose(axis, R[:, 0])
    np.testing.assert_allclose(seam, R[:, 1])
    np.testing.assert_array_equal(steps, [0.1, 0.1, 0.2])
    with pytest.raises(DatumRejected, match="outside_detector"):
        scanner_grid(g, out, {**FIT, "translation_mm": [0, 0, 20]}, G)

    with pytest.raises(DatumRejected, match="bead_intersects_inspection_grid"):
        scanner_grid(g, out, FIT, G, bead_centers=[center + 1.5 * seam], clearance=0.3)


def test_duplicate_exposure_and_stale_job(tmp_path):
    scan = tmp_path / "duplicate"
    G.save(scan.with_suffix(".geom"))
    scan.with_suffix(".f32").write_bytes(b"same exposure")
    j = tmp_path / "job"
    args = (
        scan,
        scan,
        ROOT / "fixtures/remount-fixture.json",
        ROOT / "fixtures/remount-material.grid",
        j,
    )
    kw = dict(level=0.02, threshold=0.3, radius=9.4)
    with pytest.raises(ValueError, match="distinct projection"):
        run(*args, **kw)
    assert not (j / "manifest.json").exists() and not (j / "material-loss.json").exists()
    with pytest.raises(FileExistsError):
        run(*args, **kw)
