"""Native radial wall map: sampled boundary semantics, invalids and circular sectors."""

import numpy as np
import pytest
from validate_cylinder import spec
from wall import STATUS, WALL_DTYPE, measure


def run(tmp, profiles, **kw):
    a = np.asarray(profiles, dtype="<f4")  # [height,azimuth,radius] for readable cases
    h, w, d = a.shape
    grid = spec(tmp / "管.txt", (w, h, d), (0, 0, 0), (0, 0, 1), (1, 0, 0), (2, 0.25, 0.1))
    src = tmp / "扇区.f32"
    a.transpose(2, 0, 1).tofile(src)
    out = tmp / "结果.wall"
    m = measure(src, grid, out, 0.5, **kw)
    return m, np.fromfile(out, WALL_DTYPE).reshape(h, w)


def test_classify_every_ray(tmp_path):
    cases = [
        ("measured", [0, 0, 1, 1, 0, 0, 0]),
        ("empty", [0] * 7),
        ("clipped", [1, 1, 0, 0, 0, 0, 0]),
        ("multiple", [0, 1, 0, 1, 1, 0, 0]),
        ("ambiguous", [0, 0.5, 0.5, 1, 1, 0, 0]),
        ("nonfinite", [0, 1, 1, 0, 0, np.nan, 0]),
        ("undersampled", [0, 0, 1, 0, 0, 0, 0]),
        ("measured", [0, 0.5, 1, 1, 0.5, 0, 0]),
        ("ambiguous", [0, 0.5, 0, 1, 1, 0, 0]),
        ("clipped", [0, 1, 1, 0, 0, 0, 0.5]),
        ("nonfinite", [1, 0, 0, 1, 0, 0, np.inf]),
    ]
    m, data = run(tmp_path, [[p for _, p in cases]], sector_cols=3)
    v = data[0]
    np.testing.assert_array_equal(v["status"], [STATUS.index(k) for k, _ in cases])
    assert not np.any(v["reserved"])
    invalid = v["status"] != 0
    for f in ("inner", "outer", "thickness"):
        assert np.isnan(v[f][invalid]).all()
    assert (v[0]["inner"], v[0]["outer"], v[0]["thickness"]) == (2.375, 2.875, 0.5)
    assert (v[7]["inner"], v[7]["outer"], v[7]["thickness"]) == (2.25, 3, 0.75)
    assert m["eligible_sectors"] == 0 and m["thinnest_complete_sector"] is None
    assert m["min_valid_point"] == dict(row=0, column=0, mm=0.5)
    assert m["status_counts"] == {k: sum(n == k for n, _ in cases) for k in STATUS}


def test_wrap_and_missing_axial_row(tmp_path):
    a = np.zeros((2, 9, 9))
    a[:, :, 2:7] = 1
    a[:, [8, 0, 1], 4:7] = 0
    m, _ = run(tmp_path, a, sector_cols=3)
    assert m["eligible_sectors"] == 9 and m["thinnest_complete_sector"] == dict(
        column=0, azimuth_deg=0, mean_mm=0.5
    )
    a[1, 0] = 0
    m, _ = run(tmp_path, a, sector_cols=3)
    assert m["eligible_sectors"] == 6 and m["thinnest_complete_sector"]["column"] not in {8, 0, 1}
    m, _ = run(tmp_path, np.zeros_like(a), sector_cols=3)
    assert m["thinnest_complete_sector"] is None and m["min_valid_point"] is None


@pytest.mark.parametrize(
    "kind",
    [
        "short",
        "trailing",
        "threshold",
        "nanlevel",
        "budget",
        "even",
        "wide",
        "hardlink",
        "specalias",
        "twoknots",
        "indistinct",
    ],
)
def test_failures_preserve_destination(tmp_path, kind):
    src = tmp_path / "atlas"
    grid = tmp_path / "spec"
    out = tmp_path / "out"
    out.write_bytes(b"keep")
    d = 2 if kind == "twoknots" else 5
    steps = (1e16, 0.001, 0.1) if kind == "indistinct" else (1, 0.25, 0.1)
    spec(grid, (5, 1, d), (0, 0, 0), (0, 0, 1), (1, 0, 0), steps)
    src.write_bytes(
        b"\0" * (5 * d * 4 + (1 if kind == "trailing" else -1 if kind == "short" else 0))
    )
    if kind == "hardlink":
        out.unlink()
        out.hardlink_to(src)
    if kind == "specalias":
        out = grid
    before = out.read_bytes()
    with pytest.raises(RuntimeError):
        measure(
            src,
            grid,
            out,
            {"threshold": "0.5mm", "nanlevel": "nan"}.get(kind, 0.5),
            budget=0 if kind == "budget" else 1,
            sector_cols={"even": 2, "wide": 7}.get(kind, 3),
        )
    assert out.read_bytes() == before and not list(tmp_path.glob("*.partial.*"))


def test_probe_replays_status_and_detects_changed_record(tmp_path):
    from wall_probe import probe

    a = np.array([[[0, 1, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, np.nan, 0, 0]]])
    m, _ = run(tmp_path, a)
    src = tmp_path / "扇区.f32"
    out = tmp_path / "结果.wall"
    first = probe(src, out, m, 0, 0)
    assert first["measurement"]["inner_xyz_mm"] == [2.125, 0, 0] and first["measurement"][
        "outer_xyz_mm"
    ] == [2.625, 0, 0]
    multi = probe(src, out, m, 0, 1)
    assert len(multi["crossings"]) == 4 and multi["measurement"] is None
    assert probe(src, out, m, 0, 2)["attenuation"][2] is None
    with pytest.raises(ValueError, match="outside"):
        probe(src, out, m, 0, -1)
    data = bytearray(out.read_bytes())
    data[0:8] = np.asarray(1.0, dtype="<f8").tobytes()
    out.write_bytes(data)
    with pytest.raises(ValueError, match="disagree"):
        probe(src, out, m, 0, 0)
