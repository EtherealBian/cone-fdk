import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import stripe_review as st
from compare_native import sha
from qualify_stripe_scale import dense, fixture
from region_review import analyze
from remount_loss import write_doc
from wall import WALL_DTYPE

ROOT = Path(__file__).resolve().parents[1]


def original(tmp):
    reports = ROOT / "reports"
    q = json.loads((reports / "edge-qualification.json").read_bytes())
    z = reports / "edge-review-example.zip"
    assert sha(z) == q["evidence_sha256"][z.name]
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    return q["cases"]["noisy-change"]["review_manifest_sha256"]


@pytest.mark.parametrize("rows", [1, 8, 62])
def test_signed_example_against_full_detail(tmp_path, rows):
    pin = original(tmp_path)
    d = st.review(tmp_path, pin, [[11, 46], [29, 46]], rows)
    ref = analyze(tmp_path, 9.4, pin)
    for r, s in zip(d["result"]["scenarios"], ref["scenarios"], strict=True):
        assert (
            r["observed_components"],
            r["largest_observed_cells"],
            r["observed_flagged_cells"],
            r["unknown_cells"],
        ) == (
            s["region_count"],
            s["largest_observed_cells"],
            s["observed_flagged_cells"],
            s["unknown_cells"],
        )
    assert d["work"]["payload_bytes_read"] == 209 * 360 * 41


@pytest.mark.parametrize(
    "bad",
    [
        "bad-pin",
        "trim",
        "tail-changed",
        "malformed-wall",
        "classes",
        "summary-census",
        "nominal-cross",
        "sensitive-missed",
        "wrong-format",
        "budget",
        "anchor",
    ],
)
def test_fail_closed_even_late_in_stream(tmp_path, bad):
    pin = original(tmp_path)
    mf = tmp_path / "manifest.json"
    m = json.loads(mf.read_bytes())
    anchors = [[8, 10], [11, 20]]
    rows = 7
    if bad == "bad-pin":
        pin = "0" * 64
    if bad in ("trim", "tail-changed", "malformed-wall"):
        p = tmp_path / "inspected-02.wall"
        buf = bytearray(p.read_bytes())
        if bad == "trim":
            buf = buf[:-32]
        if bad == "tail-changed":
            arr = np.frombuffer(buf, WALL_DTYPE)
            arr[-1] = arr[-2]
        if bad == "malformed-wall":
            buf[-1] = 255
        p.write_bytes(buf)
        if bad == "malformed-wall":
            m["outputs_sha256"][p.name] = sha(p)
    if bad == "classes":
        p = tmp_path / "class.u8"
        b = bytearray(p.read_bytes())
        b[-1] = 3
        p.write_bytes(b)
        m["outputs_sha256"][p.name] = sha(p)
    if bad == "summary-census":
        m["counts"]["always_below"] += 1
    if bad == "nominal-cross":
        m["nominal_by_class"]["below"]["always_below"] += 1
    if bad == "sensitive-missed":
        m["sensitive_missed_by_same_level"] += 1
    if bad == "wrong-format":
        m["format"] = "CFDK-synthetic-stripe-fixture-v1"
    if bad == "budget":
        rows = 5000
    if bad == "anchor":
        anchors[0] = [-1, 6]
    if bad not in ("bad-pin", "trim", "tail-changed"):
        write_doc(mf, m)
        pin = sha(mf)
    with pytest.raises((ValueError, KeyError)):
        st.review(tmp_path, pin, anchors, rows)


def test_algorithm_stress_is_not_a_measurement(tmp_path):
    p = tmp_path / "stress"
    fixture(p, 64, 96)
    mf = p / "manifest.json"
    doc = json.loads(mf.read_bytes())
    pin = sha(mf)
    for rows in [1, 7, 16, 64]:
        result, _ = st.scan(p, doc, pin, [[10, 0], [53, 32]], rows)
        assert result["scenarios"] == dense(64, 96)
    with pytest.raises(ValueError, match="not an edge review"):
        st.review(p, pin, [[10, 0], [53, 32]], 8)


def test_finite_but_collapsed_radial_grid_is_refused(tmp_path):
    original(tmp_path)
    mf, grid = tmp_path / "manifest.json", tmp_path / "material.grid"
    doc = json.loads(mf.read_bytes())
    tokens = grid.read_text(encoding="utf-8-sig").split()
    # Both values are finite and positive, but r0 + dr * (k-1) rounds back to r0.
    tokens[-3:-1] = ["1e100", "1e-100"]
    grid.write_bytes(" ".join(tokens).encode())
    doc["outputs_sha256"][grid.name] = sha(grid)
    write_doc(mf, doc)
    with pytest.raises(ValueError, match="bad review metadata"):
        st.review(tmp_path, sha(mf), [[0, 0], [1, 1]], 1)
