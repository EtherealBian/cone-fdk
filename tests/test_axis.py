"""Native axis: exact oblique intersections, rotated scanner coordinates and failure atomicity."""

import json

import numpy as np
import pytest
from axis_gauge import fit_axis, run
from validate_cylinder import spec
from wall import WALL_DTYPE


def sample(tmp, w=64, h=13, tilt=13, pitch=0.25, bend=0):
    # Local XY are scanner YZ; local Z is scanner X. Intersect rays with the
    # cylinder in its principal basis, independently of the native conic estimator.
    grid = spec(
        tmp / "圆柱.grid", (w, h, 201), (4, -2, 1), (1, 0, 0), (0, 1, 0), (1, 0.07, pitch)
    )
    theta, phi = np.deg2rad([tilt, -43])
    z = (np.arange(h) - (h - 1) / 2) * pitch
    u = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    origins = np.stack([0.4 + bend * (z / max(abs(z))) ** 2, -0.3 + 0 * z, 0 * z], axis=-1)
    d = np.c_[
        np.cos(np.arange(w) * 2 * np.pi / w), np.sin(np.arange(w) * 2 * np.pi / w), np.zeros(w)
    ]
    v = np.c_[np.zeros(h), np.zeros(h), z] - origins
    vp = v - (v @ u)[:, None] * u
    dp = d - (d @ u)[:, None] * u
    a = (dp * dp).sum(1)
    b = vp @ dp.T
    c = (vp * vp).sum(1)[:, None] - 7**2
    data = np.zeros((h, w), WALL_DTYPE)
    data["outer"] = (-b + np.sqrt(b * b - a * c)) / a
    data["inner"] = 3.0
    data["thickness"] = data["outer"] - data["inner"]
    return grid, data, np.roll(u, 1), tmp / "管.wall", tmp / "轴.json"


@pytest.mark.parametrize("w,h,tilt", [(32, 13, 0), (258, 14, -11), (4096, 15, 19), (36, 12, 8)])
def test_scanner_frame_and_budget_replay(tmp_path, w, h, tilt):
    g, a, u, wall, out = sample(tmp_path, w, h, tilt)
    a.tofile(wall)
    low = fit_axis(wall, g, out, 1e-9, budget=1)
    payload = out.read_bytes()
    high = fit_axis(wall, g, out, 1e-9, budget=16)
    assert payload == out.read_bytes()
    if w == 4096:
        assert low["tiles"] > high["tiles"] == 2
    for stats in [low, high]:
        assert (
            stats["passes"] == 2
            and stats["bytes_read"] == 2 * a.nbytes
            and stats["redzones_verified"]
            and stats["peak_owned_bytes"] <= stats["budget_bytes"]
        )
    fit = json.loads(payload)
    m = fit["measurement"]
    assert fit["status"] == "measured"
    np.testing.assert_allclose(m["axis_unit"], u, atol=2e-13, rtol=0)
    np.testing.assert_allclose(m["origin_xyz_mm"], [4, -1.6, 0.7], atol=2e-13, rtol=0)
    assert abs(m["outer_radius_mm"] - 7) < 1e-13 and fit["peak_residual_mm"] < 1e-13
    seam = np.array(m["seam_unit"])
    assert abs(seam @ u) < 1e-13 and abs(np.linalg.norm(seam) - 1) < 1e-13


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("missing", "incomplete"),
        ("dent", "noncylindrical"),
        ("bend", "noncylindrical"),
        ("tilt", "excess_tilt"),
        ("short", "insufficient_span"),
    ],
)
def test_no_usable_axis_on_failed_gate(tmp_path, kind, reason):
    g, a, _, wall, out = sample(
        tmp_path,
        tilt=24 if kind == "tilt" else 9,
        pitch=0.01 if kind == "short" else 0.25,
        bend=0.3 if kind == "bend" else 0,
    )
    if kind == "missing":
        a[-1, -1] = (np.nan, np.nan, np.nan, 3, 0)
    if kind == "dent":
        a["outer"][0, 0] -= 0.16
        a["thickness"][0, 0] = a["outer"][0, 0] - a["inner"][0, 0]
    a.tofile(wall)
    fit_axis(wall, g, out, 0.025)
    r = json.loads(out.read_bytes())
    assert (
        r["status"] == reason
        and r["measurement"] is None
        and r["valid_samples"] == a.size - (kind == "missing")
    )
    if reason == "noncylindrical":
        assert r["peak_residual_mm"] > 0.025


@pytest.mark.parametrize(
    "kind",
    ["odd", "few_rows", "few_columns", "budget", "limit", "late_corrupt", "truncated", "alias"],
)
def test_bad_input_preserves_output(tmp_path, kind):
    g, a, _, wall, out = sample(
        tmp_path,
        257 if kind == "odd" else 22 if kind == "few_columns" else 2048,
        h=2 if kind == "few_rows" else 35,
    )
    if kind == "late_corrupt":
        a["reserved"][-1, -1] = 1
    a.tofile(wall)
    if kind == "truncated":
        wall.write_bytes(wall.read_bytes()[:-1])
    if kind == "alias":
        out.hardlink_to(wall)
    else:
        out.write_bytes(b"old")
    before = out.read_bytes()
    with pytest.raises(RuntimeError):
        fit_axis(
            wall, g, out, "nan" if kind == "limit" else 0.025, budget=0 if kind == "budget" else 1
        )
    assert out.read_bytes() == before and not list(tmp_path.glob("*.partial.*"))


def test_job_never_reuses_stale_output(tmp_path):
    scan = tmp_path / "input"
    grid = tmp_path / "grid"
    job = tmp_path / "prior"
    job.mkdir()
    for p in (scan.with_suffix(".geom"), scan.with_suffix(".f32"), grid, job / "sections.json"):
        p.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        run(scan, grid, job, fov=32, threshold=0.02, axis_limit=0.025, bore_limit=0.025)
    assert (job / "sections.json").read_bytes() == b"unchanged" and not (
        job / "manifest.json"
    ).exists()


def test_pinned_input_denies_writers_and_replacement(tmp_path):
    from axis_gauge import pinned_reader

    p = tmp_path / "input"
    p.write_bytes(b"abcd")
    with pinned_reader(p):
        with pytest.raises(PermissionError):
            p.write_bytes(b"e")
        with pytest.raises(PermissionError):
            p.rename(tmp_path / "renamed")
        assert p.read_bytes() == b"abcd"
    p.write_bytes(b"released")


def test_independent_review_fails_on_changed_outer_points(tmp_path):
    from axis_probe import review

    g, a, _, _, _ = sample(tmp_path)
    g.rename(tmp_path / "coarse.grid")
    wall = tmp_path / "coarse.wall"
    a.tofile(wall)
    fit_axis(wall, tmp_path / "coarse.grid", tmp_path / "axis.json", 0.025)
    second = review(tmp_path)
    assert (
        second["alternative_angle_delta_deg"] < 1e-10
        and second["alternative_origin_delta_mm"] < 1e-10
    )
    out = tmp_path / "axis.json"
    original = out.read_bytes()
    for field in ("axis_unit", "origin_xyz_mm", "seam_unit"):
        doc = json.loads(original)
        m = doc["measurement"]
        m[field] = (
            (np.array(m[field]) + np.array(m["axis_unit"])).tolist()
            if field == "origin_xyz_mm"
            else (-np.array(m[field])).tolist()
        )
        out.write_text(json.dumps(doc))
        with pytest.raises(ValueError, match="axis frame"):
            review(tmp_path)
    out.write_bytes(original)
    a["outer"][1, 1] += 0.05
    a["thickness"][1, 1] += 0.05
    a.tofile(wall)
    with pytest.raises(ValueError, match="distances disagree"):
        review(tmp_path)
