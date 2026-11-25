"""Real GB10 refusal diagnostics, sealed snapshots and guarded publication, using analytic inputs."""

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import qualify_witnessed_wall as q  # noqa: E402
import witnessed_wall as w  # noqa: E402


def blob(store, data):
    k = hashlib.sha256(data).hexdigest()
    p = w.object_path(store, k)
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(data)
    return k


def save(store, d):
    return blob(store, json.dumps(d).encode())


@pytest.fixture
def fixture(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    grid = blob(store, b"CFDKCYL1 7 1 7\n0 0 0\n0 0 1\n1 0 0\n2 .25 .1\n")
    schema = {
        "format": "CFDK-body-witness-v1",
        "material_xyz_mm": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "residual_limit_mm": 0.1,
    }
    body = save(store, schema)
    cov = {"paired_valid": 7, "below_threshold": 7, "flagged": 0, "unknown": 0}
    out = {
        "material.grid": grid,
        "body-witness.json": body,
        "material-loss.json": save(
            store,
            dict(
                format="CFDK-loss-v1", threshold_mm=0.3, columns=7, rows=1, coverage=cov, runs=[]
            ),
        ),
    }
    scans = {}
    src = {
        "body-witness.json": body,
        "material.grid": grid,
        "cone-fdk.exe": blob(store, b"archived-not-executable"),
        "kernels.ptx": blob(store, b"archived-not-ptx"),
    }
    for role in w.old.ROLES:
        src[role + ".f32"] = hashlib.sha256(role.encode()).hexdigest()
        scans[role] = {
            "body_witnesses": [
                {"id": i, "status": "pass", "residual_mm": 0.0} for i in range(1, 4)
            ]
        }
        out[role + "/scanner.grid"] = grid
        out[role + "/shells.f32"] = blob(
            store, struct.pack("<49f", *(k for k in (0, 1, 1, 1, 0, 0, 0) for _ in range(7)))
        )
        out[role + "/material.wall"] = blob(
            store, struct.pack("<dddII", 2.125, 2.875, 0.75, 0, 0) * 7
        )
    doc = dict(
        format="CFDK-remount-witness-job-v1",
        status="compared",
        inputs_sha256=src,
        outputs_sha256=out,
        scans=scans,
        loss=dict(coverage=cov),
        parameters={"level_per_mm": 0.5, "loss_mm": 0.3},
    )
    return store, doc, tmp_path / "review"


@pytest.mark.parametrize("key", q.BAD)
def test_native_reaches_expected_rejection(tmp_path, key):
    q.negatives(tmp_path, key)


def test_all_ternary_profiles_fp64_and_nan_bits(tmp_path):
    a = q.analytic(tmp_path)
    assert a["records"] == 317952 and a["native"]["tiles"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "root",
        "flip_source",
        "archived_exe",
        "archived_ptx",
        "same_projection",
        "missing_projection",
        "bad_witness",
        "witness_id",
        "witness_limit",
        "wrong_cal",
        "loss_count",
        "short_wall",
        "short_atlas",
        "header",
        "withheld",
        "unwitnessed",
        "duplicate_levels",
        "nominal_absent",
        "bad_budget",
    ],
)
def test_no_native_or_result_on_bad_parent(fixture, mutation, monkeypatch):
    store, doc, dest = fixture
    src = doc["inputs_sha256"]
    out = doc["outputs_sha256"]
    levels = [0.25, 0.5, 0.75]
    budget = 1
    if mutation == "flip_source":
        w.object_path(store, out["reference/shells.f32"]).write_bytes(b"changed")
    elif mutation in ("archived_exe", "archived_ptx"):
        w.object_path(
            store, src["cone-fdk.exe" if mutation == "archived_exe" else "kernels.ptx"]
        ).write_bytes(b"changed")
    elif mutation == "same_projection":
        src["inspected.f32"] = src["reference.f32"]
    elif mutation == "missing_projection":
        src["reference.f32"] = None
    elif mutation == "bad_witness":
        doc["scans"]["reference"]["body_witnesses"][0]["status"] = "failed"
    elif mutation == "witness_id":
        doc["scans"]["inspected"]["body_witnesses"][1]["id"] = 3
    elif mutation == "witness_limit":
        doc["scans"]["reference"]["body_witnesses"][2]["residual_mm"] = 0.11
    elif mutation == "wrong_cal":
        src["body-witness.json"] = src["reference.f32"]
    elif mutation == "loss_count":
        doc["loss"]["coverage"]["unknown"] = 1
    elif mutation == "short_wall":
        out["reference/material.wall"] = blob(store, b"a")
    elif mutation == "short_atlas":
        out["reference/shells.f32"] = blob(store, b"a")
    elif mutation == "header":
        src["material.grid"] = out["material.grid"] = blob(store, b"bad grid")
    elif mutation == "withheld":
        doc["status"] = "datum_rejected"
    elif mutation == "unwitnessed":
        doc["format"] = "CFDK-remount-job-v1"
    elif mutation == "duplicate_levels":
        levels = [0.5, 0.5]
    elif mutation == "nominal_absent":
        levels = [0.25, 0.75]
    elif mutation == "bad_budget":
        budget = True
    pin = save(store, doc)
    if mutation == "root":
        w.object_path(store, pin).write_bytes(b"bad root bytes")
    monkeypatch.setattr(
        w, "native_run", lambda *a: pytest.fail("CUDA called before validating parent")
    )
    with pytest.raises((ValueError, OSError)):
        w.produce(store, pin, dest, levels, budget)
    assert not os.path.lexists(dest) and not list(dest.parent.glob(".witness-wall-*"))


@pytest.mark.parametrize(
    "outcome", ["success", "after_first", "wrong_histogram", "bad_nominal", "winner", "writer"]
)
def test_transaction(fixture, outcome, monkeypatch):
    store, doc, dest = fixture
    called = []
    real = w.native_run
    if outcome == "bad_nominal":
        doc["outputs_sha256"]["inspected/material.wall"] = blob(
            store, struct.pack("<dddII", 2.125, 2.875, 0.625, 0, 0) * 7
        )
    pin = save(store, doc)

    def intercept(*args):
        called.append(len(called))
        if outcome == "after_first" and len(called) == 2:
            raise RuntimeError("injected after first role")
        if outcome == "writer" and len(called) == 1:
            # Linux pathname DOES remain writable. Child must still use the authenticated bytes.
            w.object_path(store, doc["outputs_sha256"]["reference/shells.f32"]).write_bytes(
                b"changed after seal"
            )
        result = real(*args)
        if outcome == "wrong_histogram":
            result["status_counts_by_level"][0] = [6, 1, 0, 0, 0, 0, 0]
        if outcome == "winner" and len(called) == 2:
            dest.mkdir()
            (dest / "other").write_bytes(b"winner")
        return result

    monkeypatch.setattr(w, "native_run", intercept)
    if outcome in ("success", "writer"):
        got = w.produce(store, pin, dest, [0.25, 0.5, 0.75], 1)
        assert (
            got["counts"] == dict(always_below=7, sensitive=0, always_flagged=0, unknown=0)
            and len(called) == 2
        )
        assert (
            got["executable_products_sha256"] == w.NATIVE
            and got["parent_inputs_sha256"]["cone-fdk.exe"]
            == doc["inputs_sha256"]["cone-fdk.exe"]
        )
        assert all(w.sha(dest / k) == h for k, h in got["outputs_sha256"].items())
        with pytest.raises(ValueError):
            w.produce(store, pin, dest, [0.25, 0.5], 1)
    else:
        with pytest.raises((ValueError, OSError, RuntimeError)):
            w.produce(store, pin, dest, [0.25, 0.5, 0.75], 1)
        assert len(called) == 1 if outcome == "wrong_histogram" else len(called) == 2
        if outcome == "winner":
            assert (
                list(dest.iterdir()) == [dest / "other"]
                and (dest / "other").read_bytes() == b"winner"
            )
        else:
            assert not dest.exists()
    assert not list(dest.parent.glob(".witness-wall-*"))


@pytest.mark.parametrize("pin", [None, "../outside", "a" * 63, "X" * 64, 42])
def test_bad_object_name(tmp_path, pin):
    with pytest.raises(ValueError):
        w.object_path(tmp_path, pin)
