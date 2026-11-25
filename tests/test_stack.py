"""Oriented stacks: independent full-field oracle, stream seams and atomic I/O."""

import re

import numpy as np
import pytest
from common import Geometry, reconstruct
from spatial_oracle import backproject_points

CENTER = np.array([0.27, -0.18, 0.43])
BASIS = np.array([[0.8, 0.6, 0], [-0.36, 0.48, 0.8], [0.48, -0.64, 0.6]])


@pytest.fixture
def scan(tmp_path):
    prefix = tmp_path / "scan"
    g = Geometry(11, 9, 13, 0.67, 0.71, 92, 145, 0.213, -1)
    p = np.random.default_rng(923).normal(size=(g.a, g.v, g.u)).astype("<f4")
    g.save(prefix.with_suffix(".geom"))
    p.tofile(prefix.with_suffix(".f32"))
    return prefix, g, p


def spec(path, whd, steps):
    path.write_text(
        "CFDKSTACK1 "
        + " ".join(map(str, whd))
        + "\n"
        + "\n".join(" ".join(map(str, v)) for v in [CENTER, *steps]),
        encoding="utf-8-sig",
    )
    return path


@pytest.mark.parametrize(
    "whd,sign", [((25, 18, 6), 1), ((16, 19, 5), -1), ((1, 1, 1), -1), ((1, 17, 2), 1)]
)
def test_nrrd_and_fp64(scan, tmp_path, whd, sign):
    prefix, g, p = scan
    w, h, d = whd
    steps = BASIS * np.array([0.31, 0.24, sign * 0.47])[:, None]
    out = tmp_path / "定向层栈.nrrd"
    m = reconstruct(
        prefix,
        out,
        4096,
        12,
        2,
        4,
        stack=spec(tmp_path / "层栈.txt", whd, steps),
        stack_format="nrrd",
    )
    header, raw = out.read_bytes().split(b"\n\n", 1)
    lines = header.decode("ascii").splitlines()
    fields = dict(s.split(": ", 1) for s in lines[1:])
    assert lines[0] == "NRRD0005" and fields["type"] == "float" and fields["endian"] == "little"
    assert (
        fields["dimension"] == "3"
        and fields["encoding"] == "raw"
        and fields["space"] == "3D-right-handed"
    )
    assert fields["space units"] == '"mm" "mm" "mm"' and list(
        map(int, fields["sizes"].split())
    ) == list(whd)
    vectors = [
        np.fromstring(v, sep=",") for v in re.findall(r"\(([^()]*)\)", fields["space directions"])
    ]
    np.testing.assert_array_equal(vectors, steps)
    np.testing.assert_allclose(
        np.fromstring(fields["space origin"][1:-1], sep=","),
        CENTER - ((np.array(whd) - 1) / 2) @ steps,
        rtol=0,
        atol=2e-15,
    )
    assert m["shape_zyx"] == [d, h, w] and m["format"] == "nrrd-f32-le-stack"
    np.testing.assert_array_equal(
        [m[k] for k in ("col_step_mm", "row_step_mm", "slice_step_mm")], steps
    )
    np.testing.assert_array_equal(m["center_mm"], CENTER)
    assert len(header) + 2 == m["data_offset_bytes"] and m["output_bytes"] == out.stat().st_size
    assert m["redzones_verified"] and m["projection_passes"] == 1 and len(raw) == 4 * w * h * d
    kk, jj, ii = np.indices((d, h, w))
    ijk = np.stack([ii, jj, kk], axis=-1)
    ref, border = backproject_points(p, g, CENTER + (ijk - (np.array(whd) - 1) / 2) @ steps)
    assert border or w == 1
    got = np.frombuffer(raw, "<f4").reshape(d, h, w)
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 5e-6
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-5


def test_cross_layer_strip_reverse(scan, tmp_path):
    prefix, _, p = scan
    w, h, d = 751, 369, 5
    steps = BASIS * np.array([0.007, 0.012, 0.23])[:, None]
    out = tmp_path / "out"
    ref = None
    for budget, sign, n in [(12, 1, 64), (1, 1, 4096), (1, -1, 97)]:
        axes = steps.copy()
        axes[2] *= sign
        m = reconstruct(
            prefix, out, n, 12, budget, 4, stack=spec(tmp_path / "stack", (w, h, d), axes)
        )
        assert m["format"] == "raw-f32-le-stack" and m["data_offset_bytes"] == 0
        assert (
            m["peak_owned_bytes"] <= m["effective_budget_bytes"] == budget << 20
            and m["redzones_verified"]
        )
        rows = m["rows_per_pass"]
        passes = m["projection_passes"]
        assert (
            passes == (d * h + rows - 1) // rows
            and m["projection_bytes_read"] == passes * p.nbytes
        )
        if budget == 1:
            assert rows < h and rows % h and passes > d
        else:
            assert passes == 1
        got = np.fromfile(out, "<u4").reshape(d, h, w)[::sign]
        if ref is None:
            ref = got
        else:
            np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize(
    "text",
    [
        "CFDKPLANE1 1 1 1 0 0 0 1 0 0 0 1 0 0 0 1",
        "CFDKSTACK1 1 1 0",
        "CFDKSTACK1 1 1 4097",
        "CFDKSTACK1 1 1 1.5",
        "CFDKSTACK1 3 3 3 0 0 0 1 0 0 0 1 0",
        "CFDKSTACK1 1 1 1 0 0 0 1 0 0 0 1 0 0 0 0",
        "CFDKSTACK1 1 1 1 0 0 0 1 0 0 0 1 0 0 1 1",
        "CFDKSTACK1 1 1 1 0 0 0 1 0 0 0 1 0 1 0 1",
        "CFDKSTACK1 1 1 1 0 0 0 1 0 0 0 1 0 0 0 inf",
        "CFDKSTACK1 3 3 7 0 0 3.01 1 0 0 0 1 0 0 0 1",
        "CFDKSTACK1 1 1 1 0 0 0 1 0 0 0 1 0 0 0 1 extra",
    ],
)
def test_invalid_grid(scan, tmp_path, text):
    path = tmp_path / "spec"
    path.write_text(text)
    out = tmp_path / "old"
    out.write_bytes(b"keep")
    with pytest.raises(RuntimeError):
        reconstruct(scan[0], out, 64, 12, stack=path)
    assert out.read_bytes() == b"keep" and not list(tmp_path.glob("*.partial.*"))


def test_conflicts_aliases_and_atomic_failure(scan, tmp_path):
    prefix, _, p = scan
    path = spec(tmp_path / "spec", (5, 3, 7), BASIS)
    out = tmp_path / "out"
    out.write_bytes(b"keep")
    for kw in [
        dict(roi="0,0,0,1,1,1"),
        dict(roi_list=path),
        dict(plane=path),
        dict(stack_format="bad"),
    ]:
        with pytest.raises(RuntimeError, match="only one|stack-format"):
            reconstruct(prefix, out, 64, 12, stack=path, **kw)
    with pytest.raises(RuntimeError, match="requires --stack"):
        reconstruct(prefix, out, 64, 12, stack_format="nrrd")
    link = tmp_path / "link"
    link.hardlink_to(path)
    saved = path.read_bytes()
    with pytest.raises(RuntimeError, match="aliases"):
        reconstruct(prefix, link, 64, 12, stack=path)
    assert link.read_bytes() == saved == path.read_bytes()
    # Last batch fails after creating the NRRD header in the temporary file.
    p[-1, -1, -1] = np.nan
    p.tofile(prefix.with_suffix(".f32"))
    with pytest.raises(RuntimeError, match="non-finite"):
        reconstruct(prefix, out, 64, 12, batch=4, stack=path, stack_format="nrrd")
    assert out.read_bytes() == b"keep" and not list(tmp_path.glob("*.partial.*"))
