"""Offline release gates: wrong evidence never publishes a portable child or a speed claim."""

import copy
import json
import shutil
import zipfile

import pytest
from check_packed_regression import timing
from common import REPORT
from compare_native import sha
from region_review import analyze
from remount_loss import write_doc
from restore_packed_example import restore


def test_public_packed_overlay(tmp_path):
    root = tmp_path / "restored"
    q = json.loads((REPORT / "packed-edge-qualification.json").read_bytes())["cases"][
        "noisy-change"
    ]
    pin = sha(REPORT / "packed-edge-qualification.json")
    assert restore(REPORT, root, pin) == q["packed_edge_sha256"]
    assert sha(root / "manifest.json") == q["packed_edge_sha256"]
    with pytest.raises(FileExistsError):
        restore(REPORT, root, pin)
    r = analyze(root, 9.4, q["packed_edge_sha256"])
    write_doc(tmp_path / "region.json", r)
    assert sha(tmp_path / "region.json") == q["packed_region_sha256"]


@pytest.mark.parametrize(
    "why", ["pin", "overlay", "bad-payload", "missing", "extra", "archive-manifest", "truncated"]
)
def test_refuse_damaged_public_sample(tmp_path, why):
    reports = tmp_path / "reports"
    reports.mkdir()
    dest = tmp_path / "destination"
    overlay = "packed-edge-example-manifest.json"
    archive = "edge-review-example.zip"
    record = "packed-edge-qualification.json"
    for name in (record, overlay):
        shutil.copyfile(REPORT / name, reports / name)
    pin = sha(reports / record)
    if why == "pin":
        pin = "a" * 64
    if why == "overlay":
        (reports / overlay).write_bytes(b"{}")
    with zipfile.ZipFile(REPORT / archive) as src, zipfile.ZipFile(reports / archive, "w") as dst:
        for n in src.namelist():
            if n == "material.grid" and why == "missing":
                continue
            data = src.read(n)
            if why == "bad-payload" and n == "class.u8":
                data = bytes([data[0] ^ 1]) + data[1:]
            if why == "archive-manifest" and n == "manifest.json":
                data += b" "
            if why == "truncated" and n == "reference-01.wall":
                data = data[:-1]
            dst.writestr(n, data)
        if why == "extra":
            dst.writestr("unexpected.wall", b"")
    with pytest.raises(ValueError):
        restore(reports, dest, pin)
    assert not dest.exists() and list(tmp_path.iterdir()) == [reports]


@pytest.mark.parametrize(
    "why",
    ["order", "gpu", "time", "process count", "memory", "median", "payload", "pin", "win count"],
)
def test_benchmark_report_cannot_hide_bad_pair(why):
    q = json.loads((REPORT / "packed-edge-qualification.json").read_bytes())
    base = json.loads((REPORT / "packed-edge-ab.json").read_bytes())
    gpu = base["cases"]["three-level-review"]["runs"][0][0]["gpu"]
    assert timing(base, q, gpu)
    b = copy.deepcopy(base)
    g = b["cases"]["three-level-review"]
    row = g["runs"][0]
    if why == "order":
        row.reverse()
    elif why == "gpu":
        row[0]["gpu"] = "other"
    elif why == "time":
        row[1]["elapsed_ms"] = float("nan")
    elif why == "process count":
        row[1]["native_processes"] += 1
    elif why == "memory":
        row[1]["peak_owned_bytes"] = row[0]["peak_owned_bytes"] + 1
    elif why == "median":
        g["median_ms"]["packed"] -= 0.01
    elif why == "payload":
        del g["matched_outputs_sha256"]["class.u8"]
    elif why == "pin":
        b["companion_sha256"] = "0" * 64
    elif why == "win count":
        g["paired_wins"] -= 1
    with pytest.raises(RuntimeError):
        timing(b, q, gpu)
