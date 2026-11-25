"""No GPU required: portable maps preserve signed IDs, missing coverage, and upstream gates."""

import base64
import hashlib
import json
import re

import numpy as np
import pytest
import region_batch as rb
import region_console as rc
from compare_native import sha
from remount_loss import write_doc
from test_region_batch import prepare


def extract(p):
    return json.loads(
        re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            p.read_text("utf8"),
            re.S,
        )[1]
    )


@pytest.mark.parametrize(
    "values,kind", [([1, -1, 0, 55555, -1], "i32le"), ([9] * 12 + [-1] * 4 + [0] * 10, "rle32le")]
)
def test_lossless_labels_and_wrong_digest(values, kind):
    a = np.array(values, dtype="<i4")
    enc, b64 = rc.packed_map(a, hashlib.sha256(a.tobytes()).hexdigest())
    assert enc == kind
    raw = np.frombuffer(base64.b64decode(b64), "<i4")
    decoded = raw if enc == "i32le" else np.repeat(raw[1::2], raw[::2])
    np.testing.assert_array_equal(decoded, a)
    with pytest.raises(ValueError, match="display labels"):
        rc.packed_map(a, "0" * 64)


def test_single_html_replays_deterministically_and_escapes_text(tmp_path):
    recipe, batch, plan = prepare(tmp_path)
    p = tmp_path / "slip/manifest.json"
    m = json.loads(p.read_bytes())
    m["reason"] = "</script><untrusted>&"
    write_doc(p, m)
    plan["entries"][1]["manifest_sha256"] = sha(p)
    write_doc(recipe, plan)
    rb.run(recipe, batch)
    pin = sha(batch / "manifest.json")
    a, b = (rc.export(recipe, batch, tmp_path / f"out{i}", pin) for i in range(2))
    page = tmp_path / "out0/index.html"
    assert a == b and a["html_sha256"] == sha(page) and a["maps"] == 9
    assert "</script><untrusted>" not in page.read_text("utf8")
    e, r = extract(page)["entries"]
    assert r["reason"] == m["reason"] and "scenarios" not in r
    assert e["state"] == "incomplete_coverage" and e["scenarios"][2]["unknown_cells"] == 1
    assert any(len(s["nominal_overlap"]) > 1 for s in e["scenarios"])
    assert list(page.parent.iterdir()) == [page]
    with pytest.raises(ValueError, match="new export folder"):
        rc.export(recipe, batch, page.parent, pin)


@pytest.mark.parametrize(
    "mutation",
    [
        "batch_pin",
        "batch_result",
        "source_wall",
        "fake_summary",
        "bad_order",
        "count",
        "too_large",
        "inside_batch",
        "inside_source",
    ],
)
def test_fail_closed_without_published_page(tmp_path, mutation, monkeypatch):
    recipe, batch, _ = prepare(tmp_path)
    idx, _ = rb.run(recipe, batch)
    out = tmp_path / "out"
    m = batch / "manifest.json"
    pin = sha(m)
    if mutation == "batch_pin":
        pin = "0" * 64
    elif mutation == "batch_result":
        (batch / idx["entries"][0]["result_file"]).write_bytes(b"{}")
    elif mutation == "source_wall":
        (tmp_path / "edge/reference-02.wall").write_bytes(b"")
    elif mutation == "too_large":
        monkeypatch.setattr(rc, "MAX_PAIR_CELLS", 449)
    elif mutation == "inside_batch":
        out = batch / "child"
    elif mutation == "inside_source":
        out = tmp_path / "edge/child"
    else:
        if mutation == "fake_summary":
            idx["entries"][0]["scenario_count"] += 1
        elif mutation == "bad_order":
            idx["entries"].reverse()
        else:
            idx["counts"]["upstream_refused"] += 1
        write_doc(m, idx)
        pin = sha(m)
    with pytest.raises(ValueError):
        rc.export(recipe, batch, out, pin)
    assert not out.exists()


def test_errored_source_does_not_turn_into_measurement_or_leak_local_path(tmp_path):
    recipe, batch, _ = prepare(tmp_path)
    (tmp_path / "edge/reference-00.wall").unlink()
    rb.run(recipe, batch)
    out = tmp_path / "console"
    rc.export(recipe, batch, out, sha(batch / "manifest.json"))
    e = extract(out / "index.html")["entries"][0]
    assert set(e) == {"id", "source_sha256", "state", "error_type"}
    assert e["state"] == "input_error" and e["error_type"] == "FileNotFoundError"
