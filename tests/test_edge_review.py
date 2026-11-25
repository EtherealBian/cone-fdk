"""Finite-level decisions; intentionally small, separately constructed native shell jobs."""

import json
import shutil

import numpy as np
import pytest
from common import BUILD
from compare_native import sha
from edge_probe import probe
from edge_review import ROLES, bounds, nominal_classes, run, scenarios, thickness
from loss import inspect_loss
from remount_loss import write_doc
from validate_edges import raw_oracle
from wall import WALL_DTYPE, measure


@pytest.mark.parametrize(
    "levels", [(2,), (2, 2), (1, 3), (2, float("nan")), (2, float("inf")), (0, 2), range(1, 18)]
)
def test_bad_levels(levels):
    with pytest.raises(ValueError, match="levels"):
        scenarios(levels, 2)


def test_cross_pairs_equality_and_invalid():
    # Both matched-level comparisons of cell zero have a loss of exactly 1. Cross pairing changes it.
    a = np.array([[3, 3, 3, np.nan, 4], [4, 3, 3, 3, 4.0]])
    b = np.array([[2, 2, 3, 2, np.nan], [3, 2, 3, 2, 2.0]])
    label, lo, hi, blind = bounds(a, b, 1)
    np.testing.assert_array_equal(label, [1, 2, 0, 3, 3])
    np.testing.assert_array_equal(blind, [1, 0, 0, 0, 0])
    np.testing.assert_equal(lo, [0, 1, 0, np.nan, np.nan])
    np.testing.assert_equal(hi, [2, 1, 0, np.nan, np.nan])


@pytest.mark.parametrize(
    "record",
    [
        (1, 2, np.nan, 0, 0),
        (1, 2, 1, 0, 1),
        (np.nan, np.nan, np.nan, 7, 0),
        (1, 2, 1, 2, 0),
        (-1, 1, 2, 0, 0),
        (1, 2, 2, 0, 0),
        (1, np.inf, np.inf, 0, 0),
    ],
)
def test_corrupt_records_not_unknown(record):
    with pytest.raises(ValueError, match="malformed"):
        thickness(np.array([record], WALL_DTYPE), 1, 3)


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    folder = tmp_path_factory.mktemp("edge-parent")
    grid = folder / "material.grid"
    grid.write_text("CFDKCYL1 4 1 7\n0 0 0\n0 0 1\n1 0 0\n1 0.25 0.2\n")
    write_doc(
        folder / "body-witness.json",
        dict(
            format="CFDK-body-witness-v1",
            material_xyz_mm=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            residual_limit_mm=0.05,
        ),
    )
    scans = {
        r: dict(body_witnesses=[dict(id=i, status="pass", residual_mm=0.01) for i in (1, 2, 3)])
        for r in ROLES
    }
    for r in ROLES:
        p = folder / r
        p.mkdir()
        shutil.copyfile(grid, p / "scanner.grid")
        prof = np.tile(np.array([0, 1, 4, 4, 4, 1, 0], "<f4")[:, None], (1, 4))
        prof[:, 0] = 0  # nominal unknown in both scans
        prof[2:5, 1 if r == "reference" else 2] = 2.25  # high level invalid on just one side
        prof.tofile(p / "shells.f32")
        measure(p / "shells.f32", p / "scanner.grid", p / "material.wall", 2.0)
    lp = folder / "material-loss.json"
    inspect_loss(
        folder / "reference/material.wall",
        grid,
        folder / "inspected/material.wall",
        grid,
        lp,
        0.20,
        1.5,
        budget=16,
    )
    cov = json.loads(lp.read_bytes())["coverage"]
    assert cov == dict(paired_valid=3, unknown=1, below_threshold=2, flagged=1)
    outputs = {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob("*") if p.is_file()}
    inputs = {n: sha(BUILD / n) for n in ["cone-fdk.exe", "kernels.ptx"]}
    inputs.update({n: outputs[n] for n in ("body-witness.json", "material.grid")})
    inputs.update({"reference.f32": "a" * 64, "inspected.f32": "b" * 64})
    write_doc(
        folder / "manifest.json",
        dict(
            format="CFDK-remount-witness-job-v1",
            status="incomplete_coverage",
            loss=dict(coverage=cov),
            scans=scans,
            parameters=dict(level_per_mm=2.0, loss_mm=0.20),
            outputs_sha256=outputs,
            inputs_sha256=inputs,
        ),
    )
    return folder


def test_incomplete_parent_and_offline_replay(tmp_path, parent):
    job = tmp_path / "review"
    doc = run(parent, job, [2.5, 2.0, 1.5], parent_sha256=sha(parent / "manifest.json"))
    assert (
        doc["levels_per_mm"] == [1.5, 2.0, 2.5]
        and doc["counts"]["unknown"] == 3
        and doc["native_wall_calls"] == 4
    )
    raw_oracle(parent, job, doc)
    for col in [0, 1, 2]:
        p = probe(job, 0, col)
        assert p["classification"] == "unknown" and p["min_loss_mm"] is p["max_loss_mm"] is None
        assert sum(row["loss_mm"] is None for row in p["pairs"]) == (9 if col == 0 else 3)
    with pytest.raises(FileExistsError):
        run(parent, job, [2.0, 2.5])
    with pytest.raises(ValueError, match="coordinates"):
        probe(job, 0, -1)
    (job / "class.u8").write_bytes(b"bad")
    with pytest.raises(ValueError, match="digest"):
        probe(job, 0, 0)


@pytest.mark.parametrize(
    "mutation",
    [
        "shell",
        "wall",
        "grid",
        "parent_digest",
        "tool",
        "rejected",
        "fixture_only",
        "inspected_witness",
        "reference_witness",
        "residual",
        "null_digest",
        "nominal_coverage",
        "shifted_run",
    ],
)
def test_reject_inconsistent_jobs(parent, tmp_path, mutation):
    source = tmp_path / "parent"
    shutil.copytree(parent, source)
    m = source / "manifest.json"
    doc = json.loads(m.read_bytes())
    expected = None
    if mutation in ("shell", "wall", "grid"):
        f = {
            "shell": "reference/shells.f32",
            "wall": "inspected/material.wall",
            "grid": "material.grid",
        }[mutation]
        (source / f).write_bytes(b"bad")
    elif mutation == "parent_digest":
        expected = "0" * 64
    elif mutation == "tool":
        doc["inputs_sha256"]["kernels.ptx"] = "0" * 64
    elif mutation == "rejected":
        doc["status"] = "datum_rejected"
    elif mutation == "fixture_only":
        doc["format"] = "CFDK-remount-job-v1"
    elif mutation == "inspected_witness":
        doc["scans"]["inspected"]["body_witnesses"].pop()
    elif mutation == "reference_witness":
        doc["scans"]["reference"]["body_witnesses"][2]["status"] = "missing"
    elif mutation == "residual":
        doc["scans"]["reference"]["body_witnesses"][1]["residual_mm"] = 0.1
    elif mutation == "null_digest":
        doc["outputs_sha256"]["reference/material.wall"] = None
    else:
        cov = doc["loss"]["coverage"]
        if mutation == "nominal_coverage":
            cov.update(below_threshold=1, flagged=2)
        lp = source / "material-loss.json"
        loss = json.loads(lp.read_bytes())
        loss["coverage"] = cov
        if mutation == "shifted_run":
            loss["runs"][-1][1] = 3  # identical coverage, wrong material cell
        write_doc(lp, loss)
        doc["outputs_sha256"][lp.name] = sha(lp)
    write_doc(m, doc)
    job = tmp_path / "review"
    with pytest.raises(ValueError):
        run(source, job, [2.0, 2.5], parent_sha256=expected)
    assert not (job / "manifest.json").exists()


def test_cannot_place_child_inside_parent(parent):
    with pytest.raises(ValueError, match="outside"):
        run(parent, parent / "review", [2.0, 2.5])


@pytest.mark.parametrize(
    "runs", [[[1, 0, 1, 1]], [[0, 1, 4, 1]], [[0, 0, 1, 0]], [[0, 1, 1, 2], [0, 0, 1, 1]]]
)
def test_bad_nominal_runs(runs):
    with pytest.raises(ValueError, match="nominal loss run"):
        nominal_classes(4, 1, runs)
