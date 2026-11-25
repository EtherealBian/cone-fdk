"""Native circle fitting: truth, world frame, no silent interpolation, atomic writes and tiling."""

import json

import numpy as np
import pytest
from bore import gauge
from validate_cylinder import spec
from wall import WALL_DTYPE


def fixture(tmp, w=257, h=3):
    # Axis along X: the seam/tangent plane is world Y/Z, NOT X/Y.
    grid = spec(tmp / "圆规.grid", (w, h, 201), (4, -2, 1), (1, 0, 0), (0, 1, 0), (1, 0.06, 0.2))
    a = np.zeros((h, w), dtype=WALL_DTYPE)
    phi = 2 * np.pi * np.arange(w) / w
    for field, x, y, r in [("inner", 0.7, -0.4, 4), ("outer", -0.3, 0.25, 7)]:
        d = x * np.cos(phi) + y * np.sin(phi)
        a[field] = d + np.sqrt(r * r - x * x - y * y + d * d)
    a["thickness"] = a["outer"] - a["inner"]
    return grid, a, tmp / "管.wall", tmp / "结果.json"


@pytest.mark.parametrize("w", [32, 33, 257, 4096])
def test_exact_circles_in_rotated_scanner_frame(tmp_path, w):
    grid, a, wall, out = fixture(tmp_path, w)
    a.tofile(wall)
    stats = gauge(wall, grid, out, 1e-8, budget=1)
    assert stats["measured"] == 3 and stats["output_bytes"] == out.stat().st_size
    for j, r in enumerate(json.loads(out.read_bytes())["rows"]):
        m = r["measurement"]
        x = 4 + (j - 1) * 0.2
        for edge, xyz, d in [("inner", (x, -1.3, 0.6), 8), ("outer", (x, -2.3, 1.25), 14)]:
            np.testing.assert_allclose(m[edge]["center_xyz_mm"], xyz, rtol=0, atol=1e-12)
            assert (
                abs(m[edge]["diameter_mm"] - d) < 1e-12
                and r["residual_mm"][edge + "_peak"] < 1e-12
            )
        ecc = np.hypot(1, 0.65)
        assert (
            abs(m["eccentricity_mm"] - ecc) < 1e-12
            and abs(m["model_min_gap_mm"] - (3 - ecc)) < 1e-12
        )


def test_one_missing_column_or_local_dent_rejects_whole_row(tmp_path):
    grid, a, wall, out = fixture(tmp_path)
    a[1, -1] = (np.nan, np.nan, np.nan, 3, 0)
    a["outer"][2, 0] -= 0.15
    a["thickness"][2] = a["outer"][2] - a["inner"][2]
    a.tofile(wall)
    stats = gauge(wall, grid, out, 0.025)
    rows = json.loads(out.read_bytes())["rows"]
    assert stats["rejected"] == 2 and [r["status"] for r in rows] == [
        "measured",
        "incomplete",
        "noncircular",
    ]
    assert rows[1]["valid_columns"] == 256 and rows[1]["residual_mm"] is None
    assert rows[2]["residual_mm"]["outer_rms"] < 0.025 < rows[2]["residual_mm"]["outer_peak"]
    assert all(r["measurement"] is None for r in rows[1:])


def test_multitile_independent_of_budget(tmp_path):
    grid, a, wall, out = fixture(tmp_path, w=1025, h=83)
    a.tofile(wall)
    one = gauge(wall, grid, out, 0.02, budget=1)
    raw = out.read_bytes()
    large = gauge(wall, grid, out, 0.02, budget=16)
    assert one["tiles"] > 1 == large["tiles"] and raw == out.read_bytes()
    for run in [one, large]:
        assert run["redzones_verified"] and run["peak_owned_bytes"] <= run["budget_bytes"]
        assert run["input_bytes"] == a.nbytes and run["measured"] == 83


@pytest.mark.parametrize(
    "kind",
    [
        "nan",
        "inf",
        "bad_reserved",
        "status",
        "invalid_payload",
        "thickness",
        "wrong_grid",
        "budget",
        "too_few",
        "truncated",
        "long",
        "negative",
        "limit_suffix",
        "alias",
        "late_corruption",
    ],
)
def test_bad_input_fails_without_replacing_output(tmp_path, kind):
    grid, a, wall, out = fixture(
        tmp_path, 31 if kind == "too_few" else 257, 150 if kind == "late_corruption" else 3
    )
    if kind == "nan":
        a["inner"][0, 0] = np.nan
    if kind == "inf":
        a["outer"][0, 0] = np.inf
    if kind == "bad_reserved":
        a["reserved"][0, 0] = 1
    if kind == "status":
        a["status"][0, 0] = 99
    if kind == "invalid_payload":
        a["status"][0, 0] = 1
    if kind == "thickness":
        a["thickness"][0, 0] += 0.01
    if kind == "wrong_grid":
        a[0, 0] = (0.3, 8, 7.7, 0, 0)
    if kind == "late_corruption":
        a["reserved"][-1, -1] = 42
    a.tofile(wall)
    if kind == "truncated":
        wall.write_bytes(wall.read_bytes()[:-1])
    if kind == "long":
        with wall.open("ab") as f:
            f.write(b"!")
    if kind == "alias":
        out.hardlink_to(wall)
    else:
        out.write_bytes(b"keep")
    before = out.read_bytes()
    with pytest.raises(RuntimeError):
        gauge(
            wall,
            grid,
            out,
            {"negative": -1, "limit_suffix": "0.1mm"}.get(kind, 0.025),
            budget=0 if kind == "budget" else 1,
        )
    assert out.read_bytes() == before and not list(tmp_path.glob("*.partial.*"))
