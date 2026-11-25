"""Batch failure isolation, publication and actual interruption/restart boundaries."""

import json
from pathlib import Path

import pytest
import region_batch as rb
from compare_native import sha
from remount_loss import write_doc
from test_region_review import make_edge


def prepare(tmp):
    def row(id, kind, job, **kw):
        return dict(
            id=id, kind=kind, job=job.name, manifest_sha256=sha(job / "manifest.json"), **kw
        )

    e = make_edge(tmp / "edge")
    reject = tmp / "slip"
    reject.mkdir()
    write_doc(
        reject / "manifest.json",
        dict(
            format="CFDK-remount-witness-job-v1",
            status="datum_rejected",
            loss=None,
            reason="body_witness_failed",
            rejected_scan="inspected",
            outputs_sha256={},
        ),
    )
    recipe = tmp / "recipe.json"
    plan = dict(
        format="CFDK-region-batch-plan-v1",
        entries=[
            row("scan-A", "edge", e, area_radius_mm=4),
            row("scan-B", "witness-refusal", reject),
        ],
    )
    write_doc(recipe, plan)
    return recipe, tmp / "batch", plan


def test_interrupted_between_cases(tmp_path, monkeypatch):
    recipe, out, _ = prepare(tmp_path)
    with monkeypatch.context() as m:

        def stop(*a):
            raise KeyboardInterrupt

        m.setattr(rb, "refused", stop)
        with pytest.raises(KeyboardInterrupt):
            rb.run(recipe, out)
    assert (
        not (out / "manifest.json").exists() and len(list((out / "results").glob("*.json"))) == 1
    )
    monkeypatch.setattr(rb, "analyze", lambda *a: pytest.fail("checkpoint not reused"))
    doc, stats = rb.run(recipe, out, resume=True)
    assert stats == dict(computed=1, reused=1, input_errors=0)
    assert [e["status"] for e in doc["entries"]] == ["incomplete_coverage", "upstream_refused"]
    assert all(sha(out / e["result_file"]) == e["result_sha256"] for e in doc["entries"])
    before = (out / "manifest.json").read_bytes()
    assert rb.run(recipe, out, resume=True)[1] == dict(computed=0, reused=2, input_errors=0)
    assert before == (out / "manifest.json").read_bytes()


def test_bad_source_does_not_hide_other_jobs_or_reuse_stale(tmp_path):
    recipe, out, _ = prepare(tmp_path)
    doc, _ = rb.run(recipe, out)
    path = tmp_path / "edge/inspected-00.wall"
    old = path.read_bytes()
    path.write_bytes(b"bad")
    changed, stats = rb.run(recipe, out, resume=True)
    assert [r["status"] for r in changed["entries"]] == ["input_error", "upstream_refused"]
    assert stats == dict(computed=0, reused=1, input_errors=1)
    assert "result_file" not in changed["entries"][0]
    path.write_bytes(old)
    again, stats = rb.run(recipe, out, resume=True)
    assert again == doc and stats["reused"] == 2


def test_new_bad_pin_and_accepted_witness_not_relabelled(tmp_path):
    recipe, out, plan = prepare(tmp_path)
    plan["entries"][0]["manifest_sha256"] = "0" * 64
    p = tmp_path / "slip/manifest.json"
    m = json.loads(p.read_bytes())
    m["status"] = "compared"
    write_doc(p, m)
    plan["entries"][1]["manifest_sha256"] = sha(p)
    write_doc(recipe, plan)
    doc, stats = rb.run(recipe, out)
    assert doc["counts"]["input_error"] == 2 and stats["computed"] == 0
    assert not list((out / "results").glob("*.json"))


def test_damaged_checkpoint_aborts_without_finished_index(tmp_path):
    recipe, out, _ = prepare(tmp_path)
    doc, _ = rb.run(recipe, out)
    (out / doc["entries"][0]["result_file"]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="mismatch"):
        rb.run(recipe, out, resume=True)
    assert not (out / "manifest.json").exists()


def test_changed_recipe_build_or_active_writer(tmp_path, monkeypatch):
    recipe, out, _ = prepare(tmp_path)
    rb.run(recipe, out)
    with rb.exclusive(out), pytest.raises(RuntimeError, match="lease"):
        rb.run(recipe, out, resume=True)
    with monkeypatch.context() as m:
        m.setattr(rb, "fingerprint", lambda: {})
        with pytest.raises(ValueError, match="identity changed"):
            rb.run(recipe, out, resume=True)
    recipe.write_bytes(recipe.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="identity changed"):
        rb.run(recipe, out, resume=True)
    with pytest.raises(FileExistsError):
        rb.run(recipe, out)
    assert (
        out / "manifest.json"
    ).is_file()  # writer never admitted; previous recipe's result intact


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "unknown_field",
        "pinless",
        "bad_radius",
        "empty",
        "bad_kind",
        "alias",
        "uninitialized",
        "inside",
    ],
)
def test_plan_before_creating_output(tmp_path, mutation):
    recipe, out, plan = prepare(tmp_path)
    e = plan["entries"][0]
    if mutation == "duplicate":
        e["id"] = "SCAN-b"
    elif mutation == "unknown_field":
        e["pass"] = True
    elif mutation == "pinless":
        del e["manifest_sha256"]
    elif mutation == "bad_radius":
        e["area_radius_mm"] = True
    elif mutation == "empty":
        plan["entries"] = []
    elif mutation == "bad_kind":
        e["kind"] = "trusted"
    elif mutation == "alias":
        out = Path(tmp_path / "slip/batch")
    elif mutation == "inside":
        e["job"] = "batch/source"
    write_doc(recipe, plan)
    with pytest.raises(ValueError):
        rb.run(recipe, out, resume=mutation == "uninitialized")
    assert not out.exists()


def test_os_exit_after_commit_releases_lease(tmp_path):
    # Exit without destructors/rollback hooks. The child dies at entry 2, not on a timer.
    import subprocess
    import sys

    recipe, out, _ = prepare(tmp_path)
    driver = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import region_batch as batch
batch.refused = lambda *_: os._exit(73)
batch.run(Path(sys.argv[2]), Path(sys.argv[3]))
"""
    child = subprocess.run(
        [sys.executable, "-c", driver, str(Path(rb.__file__).parent), str(recipe), str(out)],
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 73, child.stderr
    assert not (out / "manifest.json").exists()
    log, stats = rb.run(recipe, out, resume=True)
    assert stats == dict(computed=1, reused=1, input_errors=0)
    assert log["counts"]["incomplete_coverage"] == log["counts"]["upstream_refused"] == 1
