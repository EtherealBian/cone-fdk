"""Plane geometry, detector-edge oracle, bounded row streaming and atomic rejection."""

import numpy as np
import pytest
from common import Geometry, reconstruct
from spatial_oracle import backproject_points


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    prefix = tmp_path_factory.mktemp("plane") / "scan"
    g = Geometry(11, 9, 13, 0.67, 0.71, 92.0, 145.0, 0.213, -1)
    p = np.random.default_rng(719).normal(size=(g.a, g.v, g.u)).astype("<f4")
    g.save(prefix.with_suffix(".geom"))
    p.tofile(prefix.with_suffix(".f32"))
    return prefix, g, p


def spec(path, w, h, center, col, row):
    lines = [f"CFDKPLANE1 {w} {h}", *(" ".join(map(str, v)) for v in (center, col, row))]
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


@pytest.mark.parametrize(
    "wh,col,row",
    [
        ((25, 18), (0.24, -0.18, 0), (0.084, 0.112, 0.23)),
        ((16, 23), (0, -0.24, 0.18), (0.21, 0.09, 0.12)),
        ((1, 1), (0, 1, 0), (0, 0, -1)),
    ],
)
def test_plane_fp64(scan, tmp_path, wh, col, row):
    prefix, g, p = scan
    w, h = wh
    center = (0.27, -0.18, 0.43)
    manifest = spec(tmp_path / "斜截面.txt", w, h, center, col, row)
    out = tmp_path / "截面.f32"
    meta = reconstruct(prefix, out, 4096, 12, 2, 4, plane=manifest)
    jj, ii = np.indices((h, w))
    points = (
        center
        + np.multiply.outer(ii - (w - 1) / 2, col)
        + np.multiply.outer(jj - (h - 1) / 2, row)
    )
    ref, border = backproject_points(p, g, points)
    assert border or wh == (1, 1)
    got = np.fromfile(out, "<f4").reshape(h, w)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 5e-6
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-5
    assert meta["format"] == "raw-f32-le-plane" and meta["shape_yx"] == [h, w]
    assert [meta[k] for k in ("center_mm", "col_step_mm", "row_step_mm")] == list(
        map(list, (center, col, row))
    )
    assert meta["output_bytes"] == 4 * w * h and meta["redzones_verified"] and meta["batch"] == 4
    assert meta["parent_voxel_spacing_mm"] == 12 / 4096 and "voxel_spacing_mm" not in meta
    assert "output_shape_xyz" not in meta


def test_strip_seams_and_mirroring(scan, tmp_path):
    prefix, _, _ = scan
    w, h = 769, 531
    out = tmp_path / "plane.f32"
    col = np.array([0.008, 0.004, 0])
    row = np.array([-0.002, 0.004, 0.009])
    repeat = []
    for budget, direction, n in [(8, 1, 64), (1, 1, 4096), (1, -1, 128)]:
        plane = spec(tmp_path / "plane.txt", w, h, (0.27, -0.51, -0.19), col * direction, row)
        m = reconstruct(prefix, out, n, 12, budget, 4, plane=plane)
        repeat.append(np.fromfile(out, "<u4").reshape(h, w)[:, ::direction])
        assert (m["projection_passes"] == 1) == (budget == 8)
        assert (
            m["projection_bytes_read"]
            == prefix.with_suffix(".f32").stat().st_size * m["projection_passes"]
        )
        assert m["peak_owned_bytes"] <= budget << 20 and m["redzones_verified"]
    np.testing.assert_array_equal(repeat[0], repeat[1])
    np.testing.assert_array_equal(repeat[0], repeat[2])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "CFDKPLANE2 1 1",
        "CFDKPLANE1 4097 1",
        "CFDKPLANE1 1 0",
        "CFDKPLANE1 3 3 0 0 0 0 0 0 0 1 0",
        "CFDKPLANE1 3 3 0 0 0 1 0 0 1 1 0",
        "CFDKPLANE1 3 3 5 0 0 2 0 0 0 1 0",
        "CFDKPLANE1 1 1 nan 0 0 1 0 0 0 1 0",
        "CFDKPLANE1 1 1 0 0 0 1 0 0 0 1 0 extra",
        "CFDKPLANE1 3 2.5 0 0 1 0 0 0 1 0",
    ],
)
def test_bad_plane(scan, tmp_path, text):
    manifest, dest = tmp_path / "bad", tmp_path / "previous"
    manifest.write_text(text, encoding="ascii")
    dest.write_bytes(b"keep")
    with pytest.raises(RuntimeError):
        reconstruct(scan[0], dest, 64, 12, plane=manifest)
    assert dest.read_bytes() == b"keep" and not list(tmp_path.glob("*.partial.*"))


def test_conflicts_and_hardlink(scan, tmp_path):
    manifest = spec(tmp_path / "plane.txt", 3, 2, (0, 0, 0), (1, 0, 0), (0, 1, 0))
    saved = manifest.read_bytes()
    dest = tmp_path / "result.f32"
    dest.hardlink_to(manifest)
    with pytest.raises(RuntimeError, match="aliases"):
        reconstruct(scan[0], dest, 64, 12, plane=manifest)
    for kw in [dict(roi="0,0,0,1,1,1"), dict(roi_list=manifest), dict(planner="min-passes")]:
        with pytest.raises(RuntimeError, match="only one|requires"):
            reconstruct(scan[0], dest, 64, 12, plane=manifest, **kw)
    assert dest.read_bytes() == manifest.read_bytes() == saved
