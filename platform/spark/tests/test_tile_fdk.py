"""Validate both the positive dense decision and the negative ROI / FP32 decisions."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import spark_fdk as s
import qualify_tiles as q
import build_tile_fdk as b
from spark_fdk import t


def read_guard():
    return t.json_checked(s.GUARD_ROOT / "qualification.json", s.GUARDS)


def test_real_profile_and_eligibility():
    v = s.evidence()
    best, verdict = s.attest(*v)
    assert best == "16x8x8"
    assert s.route("dense", "profiled", best, verdict) == best
    assert s.route("slab2", "profiled", best, verdict) == "original"
    assert s.route("dense", "original", best, verdict) == "original"
    eligible = s.eligible(read_guard(), best, s.GUARD_ROOT)
    assert len(eligible) == 7 and all(n.split("x")[-1] != "32" for n in eligible)


@pytest.mark.parametrize(
    "fault",
    [
        "screen_order",
        "screen_timer",
        "summary",
        "parents",
        "warmup",
        "native",
        "astra",
        "acquisition",
        "build",
        "selector",
        "memory",
        "host",
        "holdout_hash",
        "gate",
        "runtime_tile",
    ],
)
def test_profile_tamper(fault):
    p, b, plan, n, a = s.evidence()
    r = p["holdout"][0]
    if fault == "screen_order":
        p["screen"].reverse()
    if fault == "screen_timer":
        p["screen"][0]["native"]["bp_ms"] = float("inf")
    if fault == "summary":
        p["summary"]["slab2"]["recommendation"] = "selected"
    if fault == "parents":
        p["parents"]["plan"] = "0" * 64
    if fault == "warmup":
        p["warmups"].pop()
    if fault == "native":
        n["output_sha256"][0] = "3" * 64
    if fault == "astra":
        a["passed"] = 1
    if fault == "acquisition":
        p["projections"] = "a" * 64
    if fault == "build":
        b["independent_products"][1]["tile.ptx"] = "e" * 64
    if fault == "selector":
        p["selected"] = "32x4x32"
    if fault == "memory":
        r["native"]["peak_owned_bytes"] += 1
    if fault == "host":
        r["native"]["host_pinned_bytes"] -= 1
    if fault == "holdout_hash":
        r["sha256"] = "e" * 64
    if fault == "gate":
        r["native"]["redzones_verified"] = 1
    if fault == "runtime_tile":
        r["native"]["gb10_tile"] = [16, 8, 16]
    with pytest.raises(ValueError):
        s.attest(p, b, plan, n, a)


@pytest.mark.parametrize(
    "values,expected",
    [([95] * 5 + [102], True), ([95.01] * 6, False), ([90] * 4 + [120, 120], False)],
)
def test_gate_at_boundaries(values, expected):
    p, _, plan, *_ = s.evidence()
    best = p["selected"]
    rows = p["holdout"]
    for r in rows:
        if r["case"] == "dense":
            r["elapsed_ms"] = 100 if r["arm"] == "original" else values[r["block"]]
    assert t.verdict(rows, best, plan)["dense"]["process"]["gate"] is expected


@pytest.mark.parametrize(
    "issue", ["omission", "bits", "sha", "best", "san", "log", "summary", "probe", "profile"]
)
def test_guard_tamper(issue):
    g = read_guard()
    best = "16x8x8"
    if issue == "omission":
        g["comparisons"].pop()
    if issue == "bits":
        g["comparisons"][0]["different_bits_voxels"] = 1
    if issue == "sha":
        g["comparisons"][0]["sha256"] = "4" * 64
    if issue == "best":
        best = "16x8x32"
    if issue == "san":
        g["sanitizers"].pop()
    if issue == "log":
        g["sanitizers"][0]["log_sha256"] = "a" * 64
    if issue == "summary":
        g["eligible_by_bits"].append("32x4x32")
    if issue == "probe":
        g["probe_source_sha256"] = "2" * 64
    if issue == "profile":
        g["profile"] = "7" * 64
    with pytest.raises(ValueError):
        s.eligible(g, best, s.GUARD_ROOT)


@pytest.mark.parametrize(
    "file,transform,pattern",
    [
        ("main.cpp", b.host, "        auto begin"),
        ("main.cpp", b.host, "        auto mode ="),
        ("kernels.cu", b.kernel, "template <bool roi>"),
        ("kernels.cu", b.kernel, "// Shared point sampler"),
    ],
)
def test_compiler_rejects_changed_source(file, transform, pattern):
    src = (t.ROOT / "build/spark/src" / file).read_text()
    assert transform(src) != src
    with pytest.raises(ValueError):
        transform(src.replace(pattern, "/* missing */"))


@pytest.mark.parametrize("tile", list(map(t.tile_name, t.TILES)))
def test_real_odd_gpu(tile, tmp_path):
    g, p = q.inputs(tmp_path)
    targets = []
    for name in ("original", tile):
        f = tmp_path / name
        r = subprocess.run(q.command(name, "roi", g, p, f), capture_output=True, timeout=30)
        assert r.returncode == 0 and not r.stderr
        q.native_check(name, "roi", r.stdout)
        targets.append(f)
    a, v = (np.fromfile(f, "<u4") for f in targets)
    row = next(r for r in read_guard()["comparisons"] if (r["case"], r["tile"]) == ("roi", tile))
    assert (t.sha(targets[0]), t.sha(targets[1])) == (row["reference_sha256"], row["sha256"])
    assert np.count_nonzero(a != v) == row["different_bits_voxels"]
    assert (row["sha256"] == row["reference_sha256"]) == (not tile.endswith("x32"))


@pytest.mark.parametrize(
    "kind",
    [
        "unknown-case",
        "unknown-policy",
        "existing",
        "link",
        "parent-link",
        "traversal",
        "corrupt-profile",
        "wrong-device",
        "guard-refused",
    ],
)
def test_refuses_before_output(kind, tmp_path, monkeypatch):
    dest = tmp_path / "new"
    keep = tmp_path / "keep"
    keep.write_text("unchanged")
    case = "dense"
    policy = "profiled"
    if kind == "unknown-case":
        case = "screen"
    if kind == "unknown-policy":
        policy = "automatic"
    if kind == "existing":
        dest = keep
    if kind == "link":
        dest.symlink_to(keep)
    if kind == "parent-link":
        parent = tmp_path / "alias"
        parent.symlink_to(tmp_path, target_is_directory=True)
        dest = parent / "new"
    if kind == "traversal":
        dest = tmp_path / ".." / tmp_path.name / "new"
    if kind == "corrupt-profile":
        monkeypatch.setattr(s, "PROFILE", "0" * 64)
    if kind == "wrong-device":
        monkeypatch.setattr(s.subprocess, "check_output", lambda *a, **k: "Other GPU, 12.1")
    if kind == "guard-refused":
        monkeypatch.setattr(s, "GUARDS", "a" * 64)
    inventory = lambda: sorted(x.name for x in tmp_path.iterdir())
    before = inventory()
    with pytest.raises(NotADirectoryError if kind == "parent-link" else ValueError):
        s.run(case, dest, policy, True)
    assert inventory() == before and keep.read_text() == "unchanged"


@pytest.mark.parametrize(
    "case,tile,retain", [("dense", "16x8x8", True), ("slab2", "original", False)]
)
def test_actual_facade_receipts(case, tile, retain):
    root = s.M / ("tile-fdk-" + ("dense" if case == "dense" else "slab") + "-review-002")
    d = json.loads((root / "result.json").read_text())
    assert (
        d["eligibility_sha256"] == s.GUARDS
        and d["profile_sha256"] == s.PROFILE
        and d["source_sha256"] == t.sha(s.__file__)
    )
    r = d["trial"]
    t.verify(r, t.json_checked(s.FIXTURE, t.PLAN))
    assert r["tile"] == tile and d["receipt_only"] is retain
    if retain:
        assert (
            not (root / "volume.f32").exists()
            and d["retained_range"]["range_sha256"] == r["sha256"]
        )
    else:
        assert t.sha(root / "volume.f32") == r["sha256"] and d["retained_range"] is None
