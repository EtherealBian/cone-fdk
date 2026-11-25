"""Windows GPU regression: packed local output uses the unchanged parent voxel centers."""

import json

import numpy as np
import pytest
from common import reconstruct
from export_volume import export_layout, verify_viewer
from phantom import generate


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    work = tmp_path_factory.mktemp("roi")
    scan, full = work / "scan", work / "full.f32"
    generate(scan, 37, 32, 48)
    n = 97
    meta = reconstruct(scan, full, n, 7.37, 32, 7)
    assert meta["slabs"] == 1
    arr = np.fromfile(full, "<u4").reshape((n,) * 3)
    return scan, n, arr


@pytest.mark.parametrize(
    "box,budget",
    [
        ((19, 7, 11, 29, 33, 21), 32),
        ((13, 15, 1, 79, 78, 91), 1),
        ((0, 0, 0, 97, 97, 97), 32),
        ((96, 96, 96, 1, 1, 1), 32),
        ((0, 27, 90, 97, 1, 7), 32),
        ((96, 0, 39, 1, 97, 1), 32),
    ],
)
def test_exact_window(reference, tmp_path, box, budget):
    scan, n, full = reference
    x, y, z, w, h, d = box
    out = tmp_path / "direct.f32"
    m = reconstruct(scan, out, n, 7.37, budget, 7, roi=",".join(map(str, box)))
    got = np.fromfile(out, "<u4").reshape(d, h, w)
    np.testing.assert_array_equal(got, full[z : z + d, y : y + h, x : x + w])
    assert m["roi_origin_xyz"] == [x, y, z] and m["output_shape_xyz"] == [w, h, d]
    assert m["redzones_verified"] and m["peak_owned_bytes"] <= budget * 2**20
    assert out.stat().st_size == m["output_bytes"] == 4 * w * h * d
    if budget == 1:
        assert m["slabs"] > 1


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "0,0,0,1,2",
        "0,0,0,1,2,3,",
        "0,0,0,1,2,3,4",
        "-1,0,0,1,1,1",
        "0,0,0,1,0,1",
        "0,0,96,1,1,2",
        "0,97,0,1,1,1",
        "96,0,0,2,1,1",
        "1,0,0,1.5,1,1",
        "0,0,0,99999999999999,1,1",
    ],
)
def test_bad_window_preserves_destination(reference, tmp_path, spec):
    scan, n, _ = reference
    out = tmp_path / "part.f32"
    out.write_bytes(b"previous result")
    with pytest.raises(RuntimeError):
        reconstruct(scan, out, n, 7.37, 32, 7, roi=spec)
    assert out.read_bytes() == b"previous result" and not list(tmp_path.glob("*.partial.*"))


def test_physical_export(reference, tmp_path):
    scan, n, _ = reference
    out, layout = tmp_path / "window.f32", tmp_path / "grid.json"
    m = reconstruct(scan, out, n, 7.37, 32, 7, roi="19,7,11,29,33,21")
    layout.write_text(json.dumps(m), encoding="utf-16")
    page = export_layout(out, layout, tmp_path, source_note="lab <example>")
    assert "lab &lt;example&gt;" in page.read_text(encoding="utf8")
    assert verify_viewer(page, 21, size_xy=(29, 33), z0=11) == 21
    header = dict(
        line.split(" = ", 1) for line in out.with_suffix(".mhd").read_text().splitlines()
    )
    assert header["DimSize"] == "29 33 21" and header["ElementDataFile"] == out.name
    np.testing.assert_allclose(
        np.fromstring(header["Offset"], sep=" "),
        (np.array([19, 7, 11]) - 48) * (7.37 / n),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.fromstring(header["ElementSpacing"], sep=" "), [7.37 / n] * 3, rtol=1e-6
    )
    m["output_shape_xyz"][1] += 1
    layout.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="layout"):
        export_layout(out, layout, tmp_path)


def test_roi_only_budget(reference, tmp_path):
    # The parent plane alone needs 64 MiB. A single off-axis tile must not allocate it.
    scan, _, _ = reference
    out = tmp_path / "tile.f32"
    m = reconstruct(scan, out, 4096, 7.37, 1, 7, roi="2049,2053,2031,17,9,5")
    data = np.fromfile(out, "<f4")
    assert m["peak_owned_bytes"] <= 2**20 and m["slabs"] == 1
    assert data.size == 17 * 9 * 5 and np.isfinite(data).all() and np.any(data)
