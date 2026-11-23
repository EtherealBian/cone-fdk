import os
import subprocess

import numpy as np
import pytest
from common import (
    BUILD,
    Geometry,
    astra_run,
    difference,
    reconstruct,
)
from phantom import generate


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    p = tmp_path_factory.mktemp("scan") / "small"
    generate(p, 37, 32, 48)
    return p


def test_dry_help():
    assert subprocess.run([BUILD / "cone-fdk.exe", "--help"], capture_output=True).returncode == 0


@pytest.mark.parametrize(
    "kind",
    [
        "short",
        "long",
        "nan",
        "inf",
        "bad-geometry",
        "singular",
        "bad-fov",
        "no-budget",
        "alias",
        "missing",
        "negative-spacing",
        "nonuniform",
        "duplicate-option",
        "extra-data",
    ],
)
def test_reject_without_replacing(scan, tmp_path, kind):

    p = tmp_path / "copy"
    g = scan.with_suffix(".geom").read_text()
    raw = scan.with_suffix(".f32").read_bytes()
    out = tmp_path / "sentinel.f32"
    out.write_bytes(b"existing result")
    n = 32
    fov = 8
    budget = 12
    if kind == "short":
        raw = raw[:-4]
    if kind == "long":
        raw += b"0000"
    if kind in ("nan", "inf"):
        raw = raw[:-4] + np.float32(float(kind)).tobytes()
    if kind == "bad-geometry":
        g = "bad header"
    if kind == "singular":
        fov = 250
    if kind == "bad-fov":
        fov = "NaN"
    if kind == "no-budget":
        budget = 1
        n = 640
    if kind == "negative-spacing":
        g = g.replace("0.48", "-0.48", 1)
    if kind == "nonuniform":
        g = g.rsplit(" ", 1)[0] + " 0.2\n"
    if kind == "extra-data":
        g += "\njunk"
    p.with_suffix(".geom").write_text(g)
    p.with_suffix(".f32").write_bytes(raw)
    if kind == "alias":
        out = p.with_suffix(".f32")
    if kind == "missing":
        p.with_suffix(".f32").unlink()
    expected = out.read_bytes()
    if kind == "duplicate-option":
        r = subprocess.run(
            [BUILD / "cone-fdk.exe", "--n", "32", "--n", "64"], capture_output=True
        )
        assert r.returncode != 0
    else:
        with pytest.raises(RuntimeError):
            reconstruct(p, out, n, fov, budget, 7)
    assert out.read_bytes() == expected and not list(tmp_path.glob("*.partial.*"))


def test_zero_signal_and_angle_schedule(scan, tmp_path):
    p = tmp_path / "zero"
    g = Geometry(37, 32, 48, 0.48, 0.48, 120, 205, 0.77, -1)
    g.save(p.with_suffix(".geom"))
    np.zeros((g.a, g.v, g.u), "f4").tofile(p.with_suffix(".f32"))
    out = tmp_path / "zero-vol.f32"
    m = reconstruct(p, out, 33, 8, 5, 7)
    assert np.array_equal(np.fromfile(out, "f4"), np.zeros(33**3, "f4"))
    assert m["launches_bp"] == 6


def test_signed_angles_rectangular_pixels(scan, tmp_path):
    # Deliberately asymmetric detector pitch and arbitrary nonzero angular origin.
    p = tmp_path / "rev"
    g = Geometry(37, 32, 48, 0.45, 0.51, 120, 205, 0.314, -1)
    g.save(p.with_suffix(".geom"))
    import shutil

    shutil.copyfile(scan.with_suffix(".f32"), p.with_suffix(".f32"))
    x = tmp_path / "native"
    y = tmp_path / "astra"
    reconstruct(p, x, 47, 7, 8, 8)
    astra_run(p, y, 47, 7)
    assert difference(x, y, 47)["rel_l2"] < 0.01


def test_auto_reduce_batch_and_slab(scan, tmp_path):
    p = tmp_path / "split"
    q = tmp_path / "mono"
    m = reconstruct(scan, p, 97, 8, 1, 64)
    reconstruct(scan, q, 97, 8, 48, 16)
    assert m["slabs"] > 1 and m["peak_owned_bytes"] <= 1024**2 and m["batch"] < 37
    assert difference(p, q, 97)["rel_l2"] <= 1e-5


def test_unicode_paths_and_atomic_replacement(scan, tmp_path):
    import shutil

    sub = tmp_path / "扫描 零件 Ω"
    sub.mkdir()
    prefix = sub / "切片"
    shutil.copyfile(scan.with_suffix(".geom"), prefix.with_suffix(".geom"))
    shutil.copyfile(scan.with_suffix(".f32"), prefix.with_suffix(".f32"))
    out = sub / "结果.f32"
    out.write_bytes(b"old")
    m = reconstruct(prefix, out, 45, 7, 12, 7)
    assert out.stat().st_size == 45**3 * 4 and m["redzones_verified"]


def test_finite_overflow_is_not_published(scan, tmp_path):
    p = tmp_path / "huge"
    p.with_suffix(".geom").write_bytes(scan.with_suffix(".geom").read_bytes())
    np.full((37, 32, 48), np.finfo("f4").max, "f4").tofile(p.with_suffix(".f32"))
    out = tmp_path / "valid.f32"
    out.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="non-finite reconstructed"):
        reconstruct(p, out, 32, 7, 12, 7)
    assert out.read_bytes() == b"previous" and not list(tmp_path.glob("*.partial.*"))


@pytest.mark.parametrize("extension", [".geom", ".f32"])
def test_reject_hardlinked_output(scan, tmp_path, extension):
    src = scan.with_suffix(extension)
    dest = tmp_path / "different-name"
    os.link(src, dest)
    original = src.read_bytes()
    with pytest.raises(RuntimeError, match="aliases"):
        reconstruct(scan, dest, 33, 8, 8, 7)
    assert dest.read_bytes() == original and os.path.samefile(src, dest)


@pytest.mark.parametrize(
    "bits,position",
    [(0x7F800000, 0), (0xFF800000, 1), (0x7FC00123, 2), (0xFF800055, 3), (0x7F800001, 8)],
)
def test_finite_vector_lanes_and_scalar_tail(tmp_path, bits, position):
    # One nine-float view per transfer: two SSE vectors and a scalar tail.
    p = tmp_path / "bits"
    Geometry(3, 3, 3, 1, 1, 20, 32).save(p.with_suffix(".geom"))
    words = np.zeros((3, 9), dtype="<u4")
    words[-1, position] = bits
    words.tofile(p.with_suffix(".f32"))
    out = tmp_path / "stable"
    out.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="non-finite line integral"):
        reconstruct(p, out, 7, 3, 8, 1)
    assert out.read_bytes() == b"original" and not list(tmp_path.glob("*.partial.*"))


def test_pageable_staging_matches_auto(scan, tmp_path):
    a, b = tmp_path / "pageable", tmp_path / "auto"
    m = reconstruct(scan, a, 73, 7.3, 1, 7, host_memory="pageable")
    other = reconstruct(scan, b, 73, 7.3, 1, 7)
    assert m["host_pinned_bytes"] == 0 and 0 <= other["host_pinned_bytes"] <= 16 * 2**20
    assert m["slabs"] > 1 and m["peak_owned_bytes"] == other["peak_owned_bytes"]
    assert a.read_bytes() == b.read_bytes()
    with pytest.raises(RuntimeError, match="--host-memory"):
        reconstruct(scan, a, 73, 7.3, 1, 7, host_memory="typo")
