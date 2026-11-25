"""Measurement release contract; no private acquisitions, CUDA jobs or historical-test substitutions."""

import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_measurement_compat as builder
import measurement_jobs as jobs
import review_pine_budget as bench
import review_measurement_history as history

OVERLAY = "f98db1a6cd2099e1ffc1ca7e8b7c5fc5c347d1083e4bbc6adbee6d5378be6910"
ARC = ROOT / "reports/gb10/pine-budget-trials.zip"
PHYSICAL = ROOT / "examples/physical-peak-batch"


def test_overlay_roundtrip_and_drift(tmp_path):
    target = tmp_path / "modules"
    assert builder.build(target, OVERLAY) == builder.build(target) == OVERLAY
    assert not (target / "axis_gauge.py").is_symlink()
    (target / "axis_gauge.py").write_text("changed")
    with pytest.raises(ValueError, match="identical"):
        builder.build(target, OVERLAY)
    assert (target / "axis_gauge.py").read_text() == "changed"


def test_wrong_overlay_pin_never_publishes(tmp_path):
    with pytest.raises(ValueError, match="digest"):
        builder.build(tmp_path / "modules", "0" * 64)
    assert not (tmp_path / "modules").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "missing-pin",
        "renamed-source",
        "changed-grid",
        "bad-tool",
        "unlisted-output",
        "changed-output",
        "lost-manifest",
        "late-destination",
        "raw-traversal",
    ],
)
def test_job_transaction(tmp_path, monkeypatch, mutation):
    source = tmp_path / "original"
    source.write_bytes(b"first")
    grid = tmp_path / "grid"
    grid.write_text("abc")
    products = tmp_path / "bin"
    products.mkdir()
    dest = tmp_path / "result"
    keys = ("wall-pack.exe", "kernels.ptx")
    raw = {"atlas": source, "grid": grid}
    pins = {k: jobs.digest(p) for k, p in raw.items()}
    for name in keys:
        (products / name).write_bytes(name.encode())
    tools = {k: jobs.digest(products / k) for k in keys}
    monkeypatch.setattr(jobs, "TOOLS", tools)
    if mutation == "missing-pin":
        pins.pop("grid")
    if mutation == "raw-traversal":
        raw["../other"] = grid
    if mutation == "changed-grid":
        grid.write_text("wrong")
    if mutation == "bad-tool":
        (products / keys[0]).write_bytes(b"wrong")

    def core(inputs, build, out, lease, opts):
        assert set(inputs) == set(raw)
        if mutation == "renamed-source":
            source.rename(tmp_path / "previous")
            source.write_bytes(b"new")
            # A separate process reopens the alias. The unrelated original name remains editable.
            child = 'import sys;from pathlib import Path;p=Path(sys.argv[1]);assert p.read_bytes()==b"first"\ntry:\n with p.open("r+b",buffering=0) as f:f.write(b"X")\nexcept PermissionError:pass\nelse:raise RuntimeError("sealed bytes writable")'
            subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", child, str(inputs["atlas"])], check=True
            )
        out.mkdir()
        p = out / "levels.wallpack"
        p.write_bytes(b"wall")
        doc = {
            "materialized": False,
            "inputs_sha256": {k: jobs.digest(v) for k, v in inputs.items()}
            | {k: jobs.digest(build / k) for k in keys},
            "pack_sha256": jobs.digest(p),
        }
        (out / "manifest.json").write_text(json.dumps(doc))
        if mutation == "unlisted-output":
            (out / "extra").write_text("wrong")
        if mutation == "changed-output":
            p.write_bytes(b"other")
        if mutation == "lost-manifest":
            (out / "manifest.json").unlink()
        if mutation == "late-destination":
            dest.mkdir()
            (dest / "keep").write_text("untouched")
        return doc

    def run():
        return jobs.run_job("pack", core, raw, dest, products, {}, pins)

    if mutation in ("none", "renamed-source"):
        doc = run()
        assert (
            doc == json.loads((dest / "manifest.json").read_bytes())
            and (dest / "levels.wallpack").read_bytes() == b"wall"
        )
        with pytest.raises(FileExistsError):
            run()
    else:
        with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
            run()
        if mutation == "late-destination":
            assert (
                list(dest.iterdir()) == [dest / "keep"]
                and (dest / "keep").read_text() == "untouched"
            )
        else:
            assert not dest.exists()
    assert not list(tmp_path.glob(".measurement-*"))


def test_benchmark_detached():
    review = bench.review(ARC, PHYSICAL)
    assert review == json.loads((ROOT / "reports/gb10/pine-budget-review.json").read_bytes())


@pytest.mark.parametrize(
    "change",
    [
        "order",
        "missing",
        "warmup",
        "swapped-case",
        "missing-pair",
        "wrong-budget",
        "budget-not-peak",
        "kernel-not-bytes",
        "wrong-volume",
        "nonfinite",
        "negative",
        "repeated-block",
    ],
)
def test_benchmark_rejects_semantic_drift(change):
    with ZipFile(ARC) as z:
        rows = [json.loads(z.read(f"trial-{i:02d}.json")) for i in range(28)]
        order = json.loads(z.read("protocol.json"))["order"]
    r = rows[4]
    v = r["run"]
    if change == "order":
        rows[6], rows[7] = rows[7], rows[6]
    elif change == "missing":
        rows.pop()
    elif change == "warmup":
        r["block"] = -1
    elif change == "swapped-case":
        r["case"] = "pine720"
    elif change == "missing-pair":
        order.pop(13)
    elif change == "wrong-budget":
        r["budget_mib"] = 384
    elif change == "budget-not-peak":
        v["peak_owned_bytes"] = v["budget_bytes"]
    elif change == "kernel-not-bytes":
        v["projection_bytes_read"] //= 3
    elif change == "wrong-volume":
        r["sha256"] = "0" * 64
    elif change == "nonfinite":
        v["process_seconds"] = float("nan")
    elif change == "negative":
        v["process_seconds"] = -1
    else:
        rows[8] = rows[4]
    with pytest.raises(ValueError):
        bench.summarize(
            rows,
            order,
            json.loads((PHYSICAL / "pine-parent.json").read_bytes()),
            json.loads((PHYSICAL / "pine-scan.json").read_bytes()),
        )


@pytest.mark.parametrize(
    "bad",
    [
        "<testsuite/>",
        '<testsuite><testcase classname="a" name="b"><skipped/></testcase></testsuite>',
        '<testsuite><testcase classname="a" name="b"/><testcase classname="a" name="b"/></testsuite>',
    ],
)
def test_partial_or_skipped_history_rejected(bad):
    with pytest.raises(ValueError):
        history.read(bad.encode())
