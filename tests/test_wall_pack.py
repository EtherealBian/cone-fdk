"""The companion must keep canonical FP64 wall bytes, including rejection NaNs, across tiles."""

import hashlib
import os

import numpy as np
import pytest
from compare_native import sha
from validate_cylinder import spec
from wall import STATUS, measure
from wall_pack import measure_pack, run, slices


def atlas(tmp_path, w=13, h=1):
    w = max(w, 7)
    grid = spec(
        tmp_path / "圆柱.grid", (w, h, 7), (0, 0, 0), (0, 0, 1), (1, 0, 0), (2, 0.25, 0.1)
    )
    raw = np.array(
        [
            [0, 1, 1, 1, 0, 0, 0],
            [0] * 7,
            [1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 0.5, 0.5, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, np.nan, 0],
            [0, 0, 1, 0, 0, 0, 0],
        ],
        "<f4",
    )
    a = raw[np.arange(w * h) % len(raw)].T.copy()
    src = tmp_path / "剖面.f32"
    a.tofile(src)
    return src, grid


@pytest.mark.parametrize(
    "w,h,levels,budget",
    [
        (13, 1, [0.5], 1),
        (360, 51, [1 / 16, 0.5, 0.875], 1),
        (32, 2, [(i + 1) / 32 for i in range(16)], 16),
    ],
)
def test_exact_old_wall_records(tmp_path, w, h, levels, budget):
    src, grid = atlas(tmp_path, w, h)
    out = tmp_path / "证据"
    doc = run(src, grid, out, levels, budget=budget, unpack=True)
    m = doc["native"]
    assert m["atlas_payload_h2d_bytes"] == m["atlas_bytes_read"] == src.stat().st_size
    assert m["output_bytes"] == w * h * 32 * len(levels)
    assert m["kernel_launches"] == len(levels) * m["tiles"] and m["tiles"] >= (
        2 if h == 51 else 1
    )
    assert doc["pack_sha256"] == sha(out / "levels.wallpack")
    for i, v in enumerate(levels):
        p = tmp_path / f"single-{i}.wall"
        old = measure(src, grid, p, v, budget=budget)
        assert sha(p) == doc["level_sha256"][i] == sha(out / f"level-{i:02}.wall")
        assert m["status_counts_by_level"][i] == [old["status_counts"][k] for k in STATUS]
        if v == 0.5:
            assert all(m["status_counts_by_level"][i])
        assert m["peak_owned_bytes"] == old["peak_owned_bytes"]
    with pytest.raises(FileExistsError):
        run(src, grid, out, levels)


@pytest.mark.parametrize(
    "mutation",
    [
        "zero",
        "descending",
        "duplicate",
        "too_many",
        "nan",
        "negative",
        "budget",
        "short",
        "extra",
        "alias",
        "hardlink",
        "no_levels",
        "nonfinite_radius",
    ],
)
def test_native_refuses_without_replacing(tmp_path, mutation):
    src, grid = atlas(tmp_path)
    out = tmp_path / "out"
    out.write_bytes(b"keep")
    levels = [0.2, 0.5]
    budget = 1
    if mutation == "zero":
        levels = [0]
    elif mutation == "descending":
        levels.reverse()
    elif mutation == "duplicate":
        levels = [0.2, 0.2]
    elif mutation == "too_many":
        levels = list(range(1, 18))
    elif mutation == "nan":
        levels = [float("nan")]
    elif mutation == "negative":
        levels = [-1]
    elif mutation == "no_levels":
        levels = []
    elif mutation == "budget":
        budget = 0
    elif mutation == "nonfinite_radius":
        grid.write_text("CFDKCYL1 13 1 7\n0 0 0\n0 0 1\n1 0 0\n1e308 1e308 .1\n")
    elif mutation == "short":
        src.write_bytes(src.read_bytes()[:-1])
    elif mutation == "extra":
        src.write_bytes(src.read_bytes() + b"x")
    elif mutation == "hardlink":
        out.unlink()
        os.link(src, out)
    elif mutation == "alias":
        out = grid
    before = out.read_bytes()
    with pytest.raises(RuntimeError):
        measure_pack(src, grid, out, levels, budget=budget)
    assert out.read_bytes() == before and not list(tmp_path.glob("*.partial.*"))


def test_offline_slices_reject_size_and_refuse_overwrite(tmp_path):
    p = tmp_path / "pack"
    p.write_bytes(b"abcdef")
    with pytest.raises(ValueError, match="length"):
        slices(p, 4, 2)
    assert slices(p, 3, 2, tmp_path) == [hashlib.sha256(v).hexdigest() for v in (b"abc", b"def")]
    with pytest.raises(FileExistsError):
        slices(p, 3, 2, tmp_path)
