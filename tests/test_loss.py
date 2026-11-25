"""Local loss topology, missing coverage, numerical semantics and failure atomicity."""

import json

import numpy as np
import pytest
from loss import inspect_loss
from loss_oracle import cpu_labels, unpack
from validate_cylinder import spec
from wall import WALL_DTYPE


def fixture(tmp, w=29, h=7):
    grid = spec(tmp / "frame.grid", (w, h, 101), (4, -2, 1), (1, 0, 0), (0, 1, 0), (2, 0.1, 0.25))
    ref = np.zeros((h, w), WALL_DTYPE)
    ref["inner"] = 5
    ref["outer"] = 9
    ref["thickness"] = 4
    return grid, ref, ref.copy(), tmp / "baseline.wall", tmp / "管.wall", tmp / "regions.json"


def save(ref, b, a_path, b_path):
    b["thickness"] = b["outer"] - b["inner"]
    ref.tofile(a_path)
    b.tofile(b_path)


def test_seam_inclusive_cut_unknown_and_world_position(tmp_path):
    g, a, b, ap, bp, out = fixture(tmp_path)
    b["outer"][2:5, [0, 28]] = 8.75  # exact equality is flagged, peak tie in row-major order
    b["inner"][4, 28] = 5.25  # deepest cell loses material on BOTH sides
    b["outer"][5, 1] = 8.75  # diagonals must not connect
    b["outer"][6, 8] = 8.5  # axial edge
    b["outer"][2, 6] = 9.75  # thickening is not loss
    b[3, 1] = (np.nan, np.nan, np.nan, 4, 0)
    a[3, 1] = (np.nan, np.nan, np.nan, 2, 0)
    save(a, b, ap, bp)
    inspect_loss(ap, g, bp, g, out, 0.25, 9)
    doc = json.loads(out.read_bytes())
    labels = unpack(doc)
    np.testing.assert_array_equal(labels, cpu_labels(a, b, 0.25))
    assert labels[3, 1] == -20
    assert doc["coverage"] == dict(paired_valid=202, unknown=1, below_threshold=194, flagged=8)
    r, diagonal, edge = doc["regions"]
    assert [r["cells"], diagonal["cells"], edge["cells"]] == [6, 1, 1]
    assert (
        r["crosses_seam"]
        and r["touches_unknown"]
        and not r["touches_axial_edge"]
        and edge["touches_axial_edge"]
    )
    assert (
        r["rows_inclusive"] == [2, 4]
        and r["axial_extent_mm"] == 0.75
        and r["min_remaining_mm"] == 3.5
    )
    peak = r["peak"]
    assert (peak["row"], peak["column"], peak["loss_mm"]) == (4, 28, 0.5)
    assert peak["outer_recession_mm"] == peak["inner_growth_mm"] == 0.25
    t = 2 * np.pi * 28 / 29
    np.testing.assert_allclose(
        peak["inspected_outer_xyz_mm"],
        [4.25, -2 + 8.75 * np.cos(t), 1 + 8.75 * np.sin(t)],
        atol=1e-14,
    )
    assert abs(r["area_mm2"] - 6 * 2 * np.pi * 9 / 29 * 0.25) < 1e-12


@pytest.mark.parametrize("kind", ["all-missing", "all-clear", "full-ring", "isolated", "bridges"])
def test_adversarial_shapes_replay_across_tile_boundaries(tmp_path, kind):
    g, a, b, ap, bp, out = fixture(tmp_path, 257, 129)
    yy, xx = np.indices(a.shape)
    if kind == "all-missing":
        b[:] = (np.nan, np.nan, np.nan, 6, 0)
    if kind == "full-ring":
        b["inner"] += 0.5
    if kind == "isolated":
        b["inner"][((xx + yy) % 2 == 0) & (xx != 256)] += 0.5
    if kind == "bridges":
        mask = np.random.default_rng(1872).random(a.shape) < 0.57
        mask[:, 0] = 1
        mask[:, -1] = 1
        mask[60, :] = 1
        b["outer"][mask] -= 0.5
        b[61, 0] = (np.nan, np.nan, np.nan, 5, 0)
    save(a, b, ap, bp)
    small = inspect_loss(ap, g, bp, g, out, 0.25, 9, budget=1)
    raw = out.read_bytes()
    doc = json.loads(raw)
    big = inspect_loss(ap, g, bp, g, out, 0.25, 9, budget=16)
    assert big["tiles"] == 1 < small["tiles"] and raw == out.read_bytes()
    assert all(
        x["peak_owned_bytes"] <= x["budget_bytes"] and x["redzones_verified"]
        for x in (small, big)
    )
    expected = cpu_labels(a, b, 0.25)
    np.testing.assert_array_equal(unpack(doc), expected)
    ids, counts = np.unique(expected[expected > 0], return_counts=True)
    assert [(r["id"], r["cells"]) for r in doc["regions"]] == list(zip(ids, counts, strict=True))
    if kind == "all-missing":
        assert doc["coverage"]["paired_valid"] == 0 and doc["coverage"]["unknown"] == a.size
    if kind == "full-ring":
        assert len(ids) == 1 and doc["regions"][0]["peak"]["column"] == 0


@pytest.mark.parametrize(
    "kind",
    [
        "grid-center",
        "grid-seam",
        "grid-axis",
        "grid-height",
        "grid-radii",
        "grid-pitch",
        "ref-size",
        "part-size",
        "late-part-corruption",
        "bad-reference",
        "hardlink",
        "grid-alias",
        "zero-loss",
        "nan",
        "bad-radius",
        "budget",
        "bad-status",
        "bad-nan",
    ],
)
def test_refuse_bad_comparisons_without_publishing(tmp_path, kind):
    g, a, b, ap, bp, out = fixture(tmp_path, 257, 70)
    ref_grid = g
    if kind.startswith("grid-") and kind != "grid-alias":
        ref_grid = spec(
            tmp_path / "other.grid",
            (257, 69 if kind == "grid-height" else 70, 102 if kind == "grid-radii" else 101),
            (4, -2, 1.01 if kind == "grid-center" else 1),
            (-1, 0, 0) if kind == "grid-axis" else (1, 0, 0),
            (0, 0, 1) if kind == "grid-seam" else (0, 1, 0),
            (2, 0.1, 0.3 if kind == "grid-pitch" else 0.25),
        )
    if kind == "late-part-corruption":
        b["reserved"][-1, -1] = 3
    if kind == "bad-reference":
        a["thickness"][0, 0] = 9
    if kind == "bad-status":
        b["status"][-1, -1] = 8
    if kind == "bad-nan":
        a[1, 1] = (1, np.nan, np.nan, 4, 0)
    save(a, b, ap, bp)
    if kind == "ref-size":
        ap.write_bytes(ap.read_bytes()[:-1])
    if kind == "part-size":
        bp.write_bytes(bp.read_bytes() + b"!")
    if kind == "hardlink":
        out.hardlink_to(ap)
    elif kind == "grid-alias":
        out = g
    else:
        out.write_bytes(b"keep original")
    before = out.read_bytes()
    with pytest.raises(RuntimeError):
        inspect_loss(
            ap,
            ref_grid,
            bp,
            g,
            out,
            {"zero-loss": 0, "nan": "nan"}.get(kind, 0.25),
            13 if kind == "bad-radius" else 9,
            budget=0 if kind == "budget" else 1,
        )
    assert out.read_bytes() == before and not list(tmp_path.glob("*.partial.*"))
