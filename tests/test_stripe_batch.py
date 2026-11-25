import json
import subprocess
import sys
from pathlib import Path

import pytest
import stripe_batch as sb
from compare_native import sha
from remount_loss import write_doc
from test_region_batch import prepare


def recipe_at(tmp):
    recipe, out, doc = prepare(tmp)
    doc["format"] = sb.PLAN
    e = doc["entries"][0]
    # The two constructed bridges change only when reference/part use different offsets.
    mf = tmp / e["job"] / "manifest.json"
    edge = json.loads(mf.read_bytes())
    edge["sensitive_missed_by_same_level"] = 2
    write_doc(mf, edge)
    e["manifest_sha256"] = sha(mf)
    del e["area_radius_mm"]
    e.update(anchors_row_col=[[2, 1], [2, 5]], band_rows=1)
    write_doc(recipe, doc)
    return recipe, out, doc


def test_real_child_exit_after_commit_then_no_relabel(tmp_path, monkeypatch):
    recipe, out, _ = recipe_at(tmp_path)
    driver = "import sys,os;sys.path.insert(0,sys.argv[1]);import stripe_batch as b;b.refused=lambda *_:os._exit(73);b.run(sys.argv[2],sys.argv[3])"
    child = subprocess.run(
        [sys.executable, "-c", driver, str(Path(sb.__file__).parent), str(recipe), str(out)],
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 73, child.stderr
    assert not (out / "manifest.json").exists()
    clean, _ = sb.run(recipe, tmp_path / "clean")
    monkeypatch.setattr(sb, "review", lambda *_: pytest.fail("unexpected relabel"))
    log, stats = sb.run(recipe, out, resume=True)
    assert clean == log and stats == dict(
        computed=1, reused=1, revalidated_payload_bytes=50 * 209, input_errors=0
    )
    log2, stats2 = sb.run(recipe, out, resume=True)
    assert log == log2 and stats2["reused"] == 2 and stats2["computed"] == 0
    assert [r["status"] for r in log["entries"]] == ["incomplete_coverage", "upstream_refused"]
    assert "anchor_states" not in log["entries"][1]


@pytest.mark.parametrize(
    "name",
    [
        "material.grid",
        "reference-00.wall",
        "inspected-02.wall",
        "class.u8",
        "min-loss.f64",
        "max-loss.f64",
        "manifest.json",
    ],
)
def test_corrupt_source_is_not_served_from_cache(tmp_path, name):
    recipe, out, _ = recipe_at(tmp_path)
    original, _ = sb.run(recipe, out)
    p = tmp_path / "edge" / name
    keep = p.read_bytes()
    p.write_bytes(keep[:-1] + bytes([keep[-1] ^ 1]))
    bad, stats = sb.run(recipe, out, resume=True)
    assert stats["reused"] == 1 and bad["counts"]["input_error"] == 1
    assert "result_sha256" not in bad["entries"][0] and "anchor_states" not in bad["entries"][0]
    p.write_bytes(keep)
    new, stats = sb.run(recipe, out, resume=True)
    assert new == original and stats["reused"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        "recipe-format",
        "batch-format",
        "source-overlap",
        "duplicate",
        "radius",
        "zero-band",
        "bool-anchor",
        "same-anchor",
        "anchor-outside",
        "pin",
    ],
)
def test_plans_are_not_interchangeable(tmp_path, bad):
    recipe, out, d = recipe_at(tmp_path)
    e = d["entries"][0]
    if bad == "recipe-format":
        d["format"] = "CFDK-region-batch-plan-v1"
    elif bad == "batch-format":
        d["format"] = sb.INDEX
    elif bad == "source-overlap":
        e["job"] = "batch/part"
    elif bad == "duplicate":
        e["id"] = "scan-b"
    elif bad == "radius":
        e["area_radius_mm"] = 4
    elif bad == "zero-band":
        e["band_rows"] = 0
    elif bad == "bool-anchor":
        e["anchors_row_col"][0][0] = True
    elif bad == "same-anchor":
        e["anchors_row_col"] = [[0, 0], [0, 0]]
    elif bad == "anchor-outside":
        e["anchors_row_col"][0][0] = 4096
    else:
        e["manifest_sha256"] = "NO"
    write_doc(recipe, d)
    with pytest.raises(ValueError):
        sb.run(recipe, out)
    assert not out.exists()


def test_outside_smaller_grid_is_isolated_per_specimen(tmp_path):
    recipe, out, d = recipe_at(tmp_path)
    d["entries"][0]["anchors_row_col"][0] = [120, 12]
    write_doc(recipe, d)
    result, stats = sb.run(recipe, out)
    assert (
        result["counts"]["upstream_refused"] == result["counts"]["input_error"] == 1
        and stats["computed"] == 1
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "recipe-anchor",
        "recipe-band",
        "engine",
        "damaged-report",
        "writer",
        "dense-journal",
        "refused-source",
    ],
)
def test_checkpoint_identity_and_lease(tmp_path, monkeypatch, mutation):
    recipe, out, d = recipe_at(tmp_path)
    doc, _ = sb.run(recipe, out)
    if mutation == "writer":
        with sb.exclusive(out), pytest.raises(RuntimeError, match="writer lease"):
            sb.run(recipe, out, resume=True)
        return
    if mutation == "refused-source":
        p = tmp_path / "slip/manifest.json"
        p.write_bytes(p.read_bytes() + b" ")
        idx, _ = sb.run(recipe, out, resume=True)
        assert (
            idx["entries"][1]["status"] == "input_error"
            and "result_file" not in idx["entries"][1]
        )
        return
    if mutation in ("recipe-anchor", "recipe-band"):
        e = d["entries"][0]
        if mutation == "recipe-anchor":
            e["anchors_row_col"][0] = [0, 1]
        else:
            e["band_rows"] = 3
        write_doc(recipe, d)
    elif mutation == "engine":
        monkeypatch.setattr(sb, "engine", lambda: {"changed": True})
    elif mutation == "damaged-report":
        (out / doc["entries"][0]["result_file"]).write_bytes(b"{}")
    elif mutation == "dense-journal":
        (out / "striped.sqlite3").rename(out / "journal.sqlite3")
    with pytest.raises(ValueError):
        sb.run(recipe, out, resume=True)
    if mutation == "damaged-report":
        assert not (out / "manifest.json").exists()


def test_entry_bytes_match_individual_and_partition_is_per_job(tmp_path):
    recipe, out, d = recipe_at(tmp_path)
    e = d["entries"][0]
    second = dict(e, id="second", band_rows=4)
    d["entries"].append(second)
    write_doc(recipe, d)
    idx, s = sb.run(recipe, out)
    assert s["computed"] == 3
    a, b = (json.loads((out / idx["entries"][i]["result_file"]).read_bytes()) for i in (0, 2))
    assert a["result"] == b["result"] and a["work"] != b["work"]
    assert a == sb.review(tmp_path / "edge", e["manifest_sha256"], e["anchors_row_col"], 1)
    assert sha(out / idx["entries"][0]["result_file"]) == idx["entries"][0]["result_sha256"]
