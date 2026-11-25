"""Fail-closed workpiece witnesses: spatial frames, ambiguous evidence, inclusive calibrated gate."""

import hashlib
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from body_witness import calibration, inspect, locate
from common import ROOT, Geometry
from remount_datum import DatumRejected
from remount_loss import scanner_grid
from witness_probe import audit

DOC = json.loads((ROOT / "fixtures/body-witness.json").read_bytes())
FIXTURE = json.loads((ROOT / "fixtures/remount-fixture.json").read_bytes())
POINTS = np.array(FIXTURE["beads_material_xyz_mm"])


def read(doc):
    return calibration(doc, POINTS, 0.65, 0.12, 0.2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("format", "v0"),
        ("material_xyz_mm", [[0, 0, 0]] * 3),
        ("material_xyz_mm", [[-10, 0, 20], [0, 0.1, 20], [10, 0, 20]]),
        ("material_xyz_mm", [POINTS[0], *DOC["material_xyz_mm"][1:]]),
        ("material_xyz_mm", [[np.nan, 0, 4]] * 3),
        ("material_xyz_mm", []),
        ("radius_mm", 0.45),
        ("level_per_mm", 0.12),
        ("search_half_mm", 0.6),
        ("residual_limit_mm", 0.3),
        ("residual_limit_mm", 0),
        ("search_half_mm", float("inf")),
    ],
)
def test_bad_calibration(key, value):
    with pytest.raises(ValueError):
        read({**DOC, key: value})


CFG = read(DOC)
G = Geometry(360, 144, 480, 0.13, 0.25, 120, 205)
FIT = dict(rotation=np.eye(3), translation_mm=np.zeros(3))


def ball():
    k, j, i = np.indices((17, 17, 17))
    return np.where((k - 8) ** 2 + (j - 8) ** 2 + (i - 8) ** 2 <= 2.5**2, 0.105, 0.04).astype(
        "<f4"
    )


def check(box, expected=(0, 0, 0), cfg=CFG):
    return locate(box, np.zeros(3), 8, 0.2, np.array(expected), cfg)


@pytest.mark.parametrize(
    "case,status",
    [
        ("none", "missing"),
        ("extra", "ambiguous"),
        ("clip", "clipped"),
        ("tiny", "wrong_volume"),
        ("rod", "wrong_shape"),
        ("nan", "invalid_window"),
        ("flat", "zero_contrast"),
    ],
)
def test_invalid_window(case, status):
    a = ball()
    if case in ("none", "tiny", "rod"):
        a.fill(0.04)
    if case == "tiny":
        a[8, 8, 8] = 0.1
    if case == "rod":
        a[3:14, 6:9, 6:8] = 0.1
    if case == "extra":
        a[2, 2, 2] = 0.1
    if case == "clip":
        a = np.roll(a, 6, axis=2)
    if case == "nan":
        a[2, 2, 2] = float("nan")
    cfg = replace(CFG, level=float(a.max())) if case == "flat" else CFG
    assert check(a, cfg=cfg)["status"] == status


def test_level_between_adjacent_fp32_values():
    a = ball()
    # Slightly above the observed FP32 maximum: must NOT round down and count this ball.
    cfg = replace(CFG, level=np.nextafter(float(a.max()), float("inf")))
    assert check(a, cfg=cfg)["status"] == "missing"


def test_exact_limit_vs_next_float_and_no_recentering():
    a = ball()
    c = np.array(check(a)["observed_scanner_xyz_mm"])
    point = c + [0.05, 0, 0]
    err = check(a, point)["residual_mm"]
    assert check(a, point, replace(CFG, limit=err))["status"] == "pass"
    assert check(a, point, replace(CFG, limit=np.nextafter(err, 0)))["status"] == "displaced"


def test_material_frame_and_parent_coordinates():
    n = 160
    lo = 20
    z, y, x = np.ogrid[lo : n - lo, :n, :n]
    vol = np.full((n - 2 * lo, n, n), 0.04, "<f4")
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    fit = dict(rotation=rot, translation_mm=[0.4, 0.2, -0.4])
    # Put every center precisely on a scanner voxel and infer the material template.
    centers = np.array([[103, 71, 55], [54, 65, 96], [58, 112, 61]])
    target = (centers - 79.5) * 0.2
    cfg = replace(CFG, points=(target - fit["translation_mm"]) @ rot)
    for i, j, k in centers:
        vol[(x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2 <= 2.5**2] = 0.105
    rows = inspect(vol, lo, 32, G, fit, cfg)
    assert [r["status"] for r in rows] == ["pass"] * 3
    np.testing.assert_allclose([r["observed_scanner_xyz_mm"] for r in rows], target, atol=1e-12)
    for r in rows:
        i, j, k, w, h, d = r["parent_roi_xyz_nxyz"]
        assert (
            hashlib.sha256(vol[k - lo : k - lo + d, j : j + h, i : i + w].tobytes()).hexdigest()
            == r["window_sha256"]
        )


@pytest.mark.parametrize(
    "shift,expected", [(20, "window_outside_scout"), (7, "window_outside_detector")]
)
def test_window_not_visible(shift, expected):
    rows = inspect(
        np.zeros((160, 160, 160), "<f4"), 0, 32, G, {**FIT, "translation_mm": [0, 0, shift]}, CFG
    )
    assert rows[1]["status"] == expected and rows[1]["residual_mm"] is None


def test_insert_intersects_inspection_region(tmp_path):
    cfg = replace(CFG, points=np.array([[7, 0, 4.5]]))
    with pytest.raises(DatumRejected, match="witness_intersects"):
        scanner_grid(ROOT / "fixtures/remount-material.grid", tmp_path / "g", FIT, G, body=cfg)


def probe_pack(path, change):
    area = ball()
    raw = area.tobytes()
    xyz = [0, 0, 0]
    config = {**DOC, "material_xyz_mm": [xyz] * 3}
    fixture = {"locator_n": 17, "locator_fov_mm": 3.4}
    blobs = {
        k: json.dumps(v).encode()
        for k, v in [("fixture.json", fixture), ("body-witness.json", config)]
    }
    rows = [
        dict(
            **check(area),
            id=i,
            window_sha256=hashlib.sha256(raw).hexdigest(),
            predicted_scanner_xyz_mm=xyz,
            parent_roi_xyz_nxyz=[0, 0, 0, 17, 17, 17],
        )
        for i in (1, 2, 3)
    ]
    if change == "nan-coordinate":
        rows[1]["observed_scanner_xyz_mm"][0] = float("nan")
    if change == "pose":
        rows[1]["predicted_scanner_xyz_mm"] = [1, 0, 0]
    job = dict(
        format="CFDK-remount-witness-job-v1",
        inputs_sha256={k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()},
        scans={
            "inspected": dict(
                material_to_scanner=dict(rotation=np.eye(3).tolist(), translation_mm=xyz),
                body_witnesses=rows,
            )
        },
    )
    blobs["manifest.json"] = json.dumps(job).encode()
    if change == "config":
        blobs["body-witness.json"] += b" "
    blobs.update({f"insert-{i}.f32": raw for i in (1, 2, 3)})
    if change == "window":
        blobs["insert-2.f32"] += b"x"
    if change == "extra":
        blobs["other"] = b"x"
    with zipfile.ZipFile(path, "w") as z:
        for k, v in blobs.items():
            z.writestr(k, v)


@pytest.mark.parametrize("change", [None, "pose", "nan-coordinate", "config", "window", "extra"])
def test_stdlib_reviewer(tmp_path, change):
    p = tmp_path / "review.zip"
    probe_pack(p, change)
    if change is None:
        assert all(r["status"] == "pass" for r in audit(p))
    else:
        with pytest.raises(ValueError):
            audit(p)
