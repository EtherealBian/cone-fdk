"""Exercise the real GB10 dense/ROI executable, vector tail, and negative performance verdict."""

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import bench_coherent_fdk as b

M = b.ROOT.parent
BASE = b.ROOT / "build/spark"
CANDIDATE = M / "coherent-fdk-build-002/coherent-fdk"
RAW = "8ba9d436088c15daacc1705a83fcc7c49bc4f2e7b7c3d1cabdb0dc1567d9d86d"


def test_sve_guard_page(tmp_path):
    path = tmp_path / "check"
    subprocess.run(
        ["g++", "-std=c++14", "-O3", str(HERE / "fdk/check_finite_sve.cpp"), "-o", str(path)],
        check=True,
    )
    p = subprocess.run([str(path)], capture_output=True, timeout=10)
    assert p.returncode == 0 and not p.stderr
    d = json.loads(p.stdout)
    n = 5 * (d["sve_bits"] // 32) + 3
    assert d["protected_tail"] is True and d["patterns"] == 14 * n * (n + 1) // 2


def inputs(wd):
    g, p = wd / "g", wd / "p"
    g.write_text("CFDK1 13 33 37 0.7 0.8 100 150 0.0 1\n")
    i = np.arange(13 * 33 * 37, dtype=np.float64)
    (0.2 + np.cos(i * 0.003) * 0.1 + np.sin(i * 0.07) * 0.03).astype("<f4").tofile(p)
    return g, p


def launch(mode, wd, geo, data, dest, box=None, options=None):
    exe = BASE / "cone-fdk" if mode == "original" else CANDIDATE
    cmd = list(
        map(
            str,
            [
                exe,
                "--geom",
                geo,
                "--input",
                data,
                "--output",
                dest,
                "--ptx",
                BASE / "kernels.ptx",
                "--n",
                63,
                "--batch",
                8,
                "--fov",
                18,
                "--budget",
                4,
            ],
        )
    )
    if mode != "original":
        cmd += ["--gb10-transport", mode]
    if box is not None:
        cmd += ["--roi", ",".join(map(str, box))]
    if options:
        cmd += options
    return subprocess.run(cmd, capture_output=True, timeout=45, cwd=wd)


@pytest.mark.parametrize("box", [None, [1, 2, 3, 27, 31, 19], [8, 9, 17, 1, 1, 1]])
def test_odd_shape_and_view_batch(tmp_path, box):
    g, p = inputs(tmp_path)
    hashes = []
    for m in b.MODES:
        f = tmp_path / m
        rr = launch(m, tmp_path, g, p, f, box)
        assert rr.returncode == 0 and not rr.stderr
        d = json.loads(rr.stdout)
        assert d["redzones_verified"] is True and d["views"] == 13 and d["filter_batches"] == 2
        hashes.append(b.sha(f))
        if m == "shared":
            assert d["bulk_d2h_bytes"] == d["bulk_h2d_bytes"] == d["host_pinned_bytes"] == 0
    assert len(set(hashes)) == 1


@pytest.mark.parametrize("mode", ["shared", "staged"])
@pytest.mark.parametrize(
    "bad", ["short", "snan", "inf", "extra", "unsupported", "duplicate", "alias", "hugecap"]
)
def test_fails_without_output(tmp_path, mode, bad):
    g, p = inputs(tmp_path)
    f = tmp_path / "dest"
    extra = None
    if bad == "short":
        p.write_bytes(p.read_bytes()[:-1])
    if bad == "extra":
        p.write_bytes(p.read_bytes() + b"x")
    if bad in ("snan", "inf"):
        with p.open("r+b") as w:
            w.seek(-4, 2)
            w.write(struct.pack("<I", 0x7F800001 if bad == "snan" else 0xFF800000))
    if bad == "unsupported":
        extra = ["--roi-list", "missing"]
    if bad == "duplicate":
        extra = ["--gb10-transport", mode]
    if bad == "alias":
        f = p
    if bad == "hugecap":
        extra = ["--budget", "769"]  # duplicate must also fail before consumption
    before = {k.name: b.sha(k) for k in tmp_path.iterdir()}
    pp = launch(mode, tmp_path, g, p, f, options=extra)
    assert pp.returncode and not pp.stdout and pp.stderr
    assert before == {k.name: b.sha(k) for k in tmp_path.iterdir()}


def test_physical_negative_verdict():
    profile = b.json_checked(M / "coherent-fdk-bench-002/crossover.json", RAW)
    plan = b.json_checked(b.ROOT / "fixtures/walnut512-slab-replay-spark-v1.json", b.PLAN)
    assert b.summary(profile["samples"], plan) == profile["summary"]
    assert all(r["preferred"] == "original" for r in profile["summary"].values())


@pytest.mark.parametrize(
    "fault", ["order", "rows", "redzones", "length", "output", "shared_count", "timing", "band"]
)
def test_corrupt_trial(fault):
    p = b.json_checked(M / "coherent-fdk-bench-002/crossover.json", RAW)
    r = p["samples"]
    plan = b.json_checked(b.ROOT / "fixtures/walnut512-slab-replay-spark-v1.json", b.PLAN)
    v = next(q for q in r if q["mode"] == "shared")
    if fault == "order":
        r.reverse()
    if fault == "rows":
        r.pop()
    if fault == "redzones":
        v["native"]["redzones_verified"] = 1
    if fault == "length":
        v["native"]["output_bytes"] -= 4
    if fault == "output":
        v["sha256"] = "0" * 64
    if fault == "shared_count":
        v["native"]["shared_alloc_bytes"] -= 512
    if fault == "timing":
        v["elapsed_ms"] = float("nan")
    if fault == "band":
        v["native"]["batch"] = 8
    with pytest.raises(ValueError):
        b.summary(r, plan)
