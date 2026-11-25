import base64
import copy
import hashlib
import itertools
import json
import struct
from collections import deque
from pathlib import Path

import anchor_audit as aa
import pytest

ROOT = Path(__file__).resolve().parents[1]


def neighbours(i, w, n):
    y, x = divmod(i, w)
    return [
        v
        for v in range(n)
        if (v // w == y and (v % w - x) % w in {1, w - 1})
        or (v % w == x and abs(y - v // w) == 1)
    ]


def distances(allowed, w, a):
    if a not in allowed:
        return {}
    lengths = {a: 0}
    q = deque([a])
    while q:
        i = q.popleft()
        for j in neighbours(i, w, 6):
            if j in allowed and j not in lengths:
                q.append(j)
                lengths[j] = lengths[i] + 1
    return lengths


def test_exhaustive_all_blind_assignments_and_shortest_certificates():
    # 729 three-valued cylinders. Independently enumerate every completion, not another 0-1 BFS.
    for values in itertools.product((-1, 0, 1), repeat=6):
        pos = {i for i, v in enumerate(values) if v > 0}
        blind = [i for i, v in enumerate(values) if v < 0]
        labels = list(values)
        for i in pos:
            labels[i] = min(distances(pos, 3, i)) + 1
        for a, b in ((0, 4), (0, 2)):
            result = aa.route(labels, 3, a, b)
            if 0 in (labels[a], labels[b]):
                assert result["state"] == "anchor_below_gate" and result["path"] is None
                continue
            found = None
            for k in range(len(blind) + 1):
                choices = [
                    ds[b] + 1
                    for add in itertools.combinations(blind, k)
                    if b in (ds := distances(pos | set(add), 3, a))
                ]
                if choices:
                    found = (k, min(choices))
                    break
            if found is None:
                assert result["state"] == "separated_on_grid" and result["path"] is None
                continue
            assert (result["min_unknown_cells"], result["path"]["cells"]) == found
            assert result["state"] == ("unresolved_blind_path" if found[0] else "observed_link")
            at = a
            path = [a]
            for direction, steps in result["path"]["direction_runs"]:
                for _ in range(steps):
                    y, x = divmod(at, 3)
                    dy, dx = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}[direction]
                    at = (y + dy) * 3 + (x + dx) % 3
                    path.append(at)
            assert (
                at == b
                and all(v >= 0 and v < 6 and labels[v] != 0 for v in path)
                and len(set(path)) == len(path)
            )
            assert (
                sum(labels[v] < 0 for v in path) == found[0]
                and hashlib.sha256(struct.pack("<" + "i" * len(path), *path)).hexdigest()
                == result["path"]["row_major_i32le_sha256"]
            )


def test_no_axial_wrap_and_unknown_elsewhere_does_not_hide_provable_link():
    assert aa.route([1, 0, 1, 0, 0, 0, 7, -1, 0], 3, 0, 2)["state"] == "observed_link"
    assert aa.route([1, 0, 1, 0, 0, 0, 7, -1, 0], 3, 0, 6)["state"] == "separated_on_grid"
    with pytest.raises(ValueError, match="contradict"):
        aa.route([1, 0, 3], 3, 0, 2)


def specimen():
    labels = [1, 0, 1, -1, 0, 0]
    raw = struct.pack("<6i", *labels)
    s = dict(
        pair=[0, 0],
        pixels=["i32le", base64.b64encode(raw).decode()],
        unknown_cells=1,
        observed_flagged_cells=2,
        labels_i32le_sha256=hashlib.sha256(raw).hexdigest(),
    )
    e = dict(
        id="demo",
        state="incomplete_coverage",
        shape=[2, 3],
        levels=[0.02],
        nominal=[0, 0],
        gate=0.30,
        scenarios=[s],
        source_sha256="a" * 64,
        result_sha256="b" * 64,
    )
    return dict(format="CFDK-region-console-v1", entries=[e])


def page(tmp, d):
    path = tmp / "console.html"
    data = b"<html>" + aa.TAG + json.dumps(d).encode() + b"</script></html>"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "change,match",
    [
        ("refusal", "no measured"),
        ("pairs", "global pair"),
        ("oversize", "grid size"),
        ("coverage", "coverage disagrees"),
        ("hash", "label map"),
        ("rle-bomb", "RLE run"),
        ("truncated", "truncated"),
        ("nominal", "nominal"),
        ("nan", "loss gate"),
        ("dup-entry", "duplicate entry"),
        ("out-of-range", "label ID"),
    ],
)
def test_damaged_inputs_not_scored(tmp_path, change, match):
    d = specimen()
    e = d["entries"][0]
    s = e["scenarios"][0]
    if change == "refusal":
        e["state"] = "upstream_refused"
    if change == "pairs":
        e["scenarios"] += copy.deepcopy(e["scenarios"])
    if change == "oversize":
        e["shape"] = [2**28, 3]
    if change == "coverage":
        s["unknown_cells"] = 0
    if change == "hash":
        s["labels_i32le_sha256"] = "0" * 64
    if change in ("rle-bomb", "truncated"):
        s["pixels"] = [
            "rle32le",
            base64.b64encode(
                struct.pack("<ii", 2**30 if change == "rle-bomb" else 2, -1)
            ).decode(),
        ]
    if change == "nominal":
        e["nominal"] = [1, 1]
    if change == "nan":
        e["gate"] = float("nan")
    if change == "dup-entry":
        d["entries"].append(copy.deepcopy(e))
    if change == "out-of-range":
        raw = struct.pack("<6i", 1, 0, 1, -8, 0, 0)
        s["labels_i32le_sha256"] = hashlib.sha256(raw).hexdigest()
        s["pixels"] = ["i32le", base64.b64encode(raw).decode()]
    p, pin = page(tmp_path, d)
    with pytest.raises(ValueError, match=match):
        aa.audit(p, pin, "demo", [[0, 0], [0, 2]])


@pytest.mark.parametrize("anchors", [[[0, 0], [0, 0]], [[-1, 0], [0, 2]], [[0, 0], [1, 3]]])
def test_material_coordinates_are_not_clamped(tmp_path, anchors):
    p, pin = page(tmp_path, specimen())
    with pytest.raises(ValueError, match="distinct"):
        aa.audit(p, pin, "demo", anchors)


def test_pin_and_publish_no_overwrite(tmp_path):
    p, pin = page(tmp_path, specimen())
    anchors = [[0, 0], [0, 2]]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        aa.audit(p, "0" * 64, "demo", anchors)
    d = aa.audit(p, pin, "demo", anchors)
    out = tmp_path / "audit.json"
    aa.publish(out, d)
    assert json.loads(out.read_bytes()) == d
    with pytest.raises(FileExistsError):
        aa.publish(out, {"not": "same"})
    assert json.loads(out.read_bytes()) == d and len(list(tmp_path.iterdir())) == 2


def test_published_critical_and_invalid_coverage():
    reports = ROOT / "reports"
    bundle = json.loads((reports / "bridge-bundle.json").read_bytes())
    orig = json.loads((reports / "bridge-result.json").read_bytes())
    br = reports / "bridge-console.html"
    for name, c in orig["cases"].items():
        d = aa.audit(br, bundle["console"]["html_sha256"], name, [[11, 46], [29, 46]])
        for a, b in zip(d["scenarios"], c["rows"], strict=True):
            assert (
                a["pair"] == b["pair"]
                and a["labels_i32le_sha256"] == b["labels_i32le_sha256"]
                and (a["state"] == "observed_link") == b["connected"]
                and a["anchor_label_ids"] == b["anchor_ids"]
            )
    ref = json.loads((reports / "region-console-browser.json").read_bytes())["html_sha256"]
    p = reports / "region-console.html"
    d = aa.audit(p, ref, "invalid-edge-control", [[11, 223], [12, 218]])
    assert d["counts"]["unresolved_blind_path"] > 0
    with pytest.raises(ValueError, match="no measured"):
        aa.audit(p, ref, "slide", [[11, 223], [12, 218]])


@pytest.mark.parametrize(
    "change",
    ["endpoint", "path-digest", "steps", "direction", "blind-count", "state", "map-digest"],
)
def test_independent_native_verifier_rejects_broken_path(tmp_path, change):
    from qualify_anchors import check_row

    p, pin = page(tmp_path, specimen())
    pts = [[0, 0], [0, 2]]
    r = aa.audit(p, pin, "demo", pts)["scenarios"][0]
    if change == "endpoint":
        r["anchor_label_ids"][0] = -1
    if change == "path-digest":
        r["path"]["row_major_i32le_sha256"] = "1" * 64
    if change == "steps":
        r["path"]["cells"] += 1
    if change == "direction":
        r["path"]["direction_runs"] = [["E", 1]]
    if change == "blind-count":
        r["min_unknown_cells"] = 1
    if change == "state":
        r["state"] = "separated_on_grid"
    if change == "map-digest":
        r["labels_i32le_sha256"] = "2" * 64
    with pytest.raises(ValueError):
        check_row(r, [[1, 0, 1], [-1, 0, 0]], pts)
