"""Native cylindrical reconstruction: non-affine address, clipping and output protection."""

import numpy as np
import pytest
from common import Geometry, reconstruct
from spatial_oracle import backproject_points


@pytest.fixture
def scan(tmp_path):
    prefix = tmp_path / "scan"
    g = Geometry(13, 12, 9, 0.73, 0.68, 92, 145, 0.32, -1)
    p = np.random.default_rng(191).normal(size=(g.a, g.v, g.u)).astype("<f4")
    g.save(prefix.with_suffix(".geom"))
    p.tofile(prefix.with_suffix(".f32"))
    return prefix, g, p


def manifest(path, dims, center, axis, seam, st):
    path.write_text(
        "CFDKCYL1 " + " ".join(str(x) for v in (dims, center, axis, seam, st) for x in v),
        encoding="utf-8-sig",
    )
    return path


@pytest.mark.parametrize("w,h,d,sign", [(25, 16, 4, 1), (24, 15, 1, -1), (3, 1, 1, 1)])
def test_tilted_cylinder(scan, tmp_path, w, h, d, sign):
    prefix, g, p = scan
    center = np.array([0.3, -0.8, 0.6])
    axis = sign * np.array([0.8, 0, 0.6])
    seam = np.array([0.36, 0.8, -0.48])
    v = np.cross(axis, seam)
    spec = manifest(
        tmp_path / "曲面.txt", (w, h, d), center, axis, seam, (2, 0.3 if d > 1 else 0, 0.24)
    )
    out = tmp_path / "图"
    m = reconstruct(prefix, out, 512, 12, 2, 4, cylinder=spec)
    assert (
        m["format"] == "raw-f32-le-cylinder"
        and m["shape_rha"] == [d, h, w]
        and m["duplicate_endpoint"] is False
        and m["data_offset_bytes"] == 0
    )
    np.testing.assert_allclose(
        [m["axis_unit"], m["seam_unit"], m["tangent_unit"]], [axis, seam, v], atol=1e-15
    )
    k, j, i = np.indices((d, h, w))[..., None]
    theta = i * 2 * np.pi / w
    pts = (
        center
        + (j - (h - 1) / 2) * 0.24 * axis
        + (2 + k * 0.3) * (seam * np.cos(theta) + v * np.sin(theta))
    )
    ref, border = backproject_points(p, g, pts)
    assert border or w == 3
    got = np.fromfile(out, "<f4").reshape(d, h, w)
    delta = got - ref
    assert (
        np.linalg.norm(delta) / np.linalg.norm(ref) < 5e-6
        and abs(delta).max() / abs(ref).max() < 1e-5
    )


@pytest.mark.parametrize(
    "dims,axis,seam,steps",
    [
        ((2, 1, 1), (0, 0, 1), (1, 0, 0), (1, 0, 0.2)),
        ((8, 2.5, 1), (0, 0, 1), (1, 0, 0), (1, 0, 0.2)),
        ((8, 3, 2), (0, 0, 1), (1, 0, 0), (1, 0, 0.2)),
        ((8, 3, 1), (0, 0, 1), (1, 0, 0), (0, 0, 0.2)),
        ((8, 3, 1), (0, 0, 1), (0, 0, 1), (1, 0, 0.2)),
        ((8, 3, 1), (0, 0, 2), (1, 0, 0), (1, 0, 0.2)),
        ((8, 3, 1), (0, 0, 1), (1, 0, 0), (1, 0, "nan")),
        ((3, 1, 1), (0, 0, 1), (0, 1, 0), (6.1, 0, 0.2)),
        ((8, 3, 1), (0, 0, 1), (1, 0, 0), (1, 0, 0)),
        ((8, 3, 2), (0, 0, 1), (1, 0, 0), (2, -1, 0.2)),
    ],
)
def test_invalid(scan, tmp_path, dims, axis, seam, steps):
    path = manifest(tmp_path / "spec", dims, (0, 0, 0), axis, seam, steps)
    out = tmp_path / "old"
    out.write_bytes(b"keep")
    with pytest.raises(RuntimeError):
        reconstruct(scan[0], out, 16, 12, cylinder=path)
    assert out.read_bytes() == b"keep"


def test_atomic(scan, tmp_path):
    prefix, _, p = scan
    spec = manifest(tmp_path / "spec", (17, 3, 1), (0, 0, 0), (0, 0, 1), (1, 0, 0), (2, 0, 0.2))
    saved = spec.read_bytes()
    link = tmp_path / "alias"
    link.hardlink_to(spec)
    with pytest.raises(RuntimeError, match="aliases"):
        reconstruct(prefix, link, 16, 12, cylinder=spec)
    assert spec.read_bytes() == saved
    out = tmp_path / "old"
    out.write_bytes(b"keep")
    for kw in [
        dict(stack=spec),
        dict(plane=spec),
        dict(roi_list=spec),
        dict(roi="0,0,0,1,1,1"),
        dict(stack_format="nrrd"),
    ]:
        with pytest.raises(RuntimeError, match="only one|requires --stack"):
            reconstruct(prefix, out, 16, 12, cylinder=spec, **kw)
    p[-1, -1, -1] = np.inf
    p.tofile(prefix.with_suffix(".f32"))
    with pytest.raises(RuntimeError, match="non-finite"):
        reconstruct(prefix, out, 16, 12, batch=4, cylinder=spec)
    assert out.read_bytes() == b"keep" and not list(tmp_path.glob("*.partial.*"))
