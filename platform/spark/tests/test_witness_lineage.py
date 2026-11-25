"""Lineage replay rejects a different parent even when every numeric payload is unchanged."""

import copy
import sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import witness_lineage as l

A, B, C, D, E, F = [c * 64 for c in "abcdef"]


@pytest.mark.parametrize("bad", ["", True, "1" * 63, "A" * 64, "../" + A, None, 1])
def test_pin(bad):
    with pytest.raises(ValueError):
        l.pin(bad)


def test_path_and_bounds(tmp_path):
    p = tmp_path / "doc"
    p.write_bytes(b"hello")
    h = l.sha(p.read_bytes())
    assert l.read(p, h, 5) == b"hello"
    sy = tmp_path / "sy"
    sy.symlink_to(p)
    for path, pin, cap in [(sy, h, 5), (p, A, 5), (p, h, 4), (tmp_path, h, 50)]:
        with pytest.raises(ValueError):
            l.read(path, pin, cap)


def edges():
    old = dict(
        format="CFDK-edge-review-v1",
        inputs_sha256={"manifest.json": A, "wall-pack.exe": D, "material.grid": F},
        **{k: [1] for k in l.META},
    )
    new = {k: copy.deepcopy(v) for k, v in old.items() if k != "inputs_sha256"}
    new.update(
        producer_schema="CFDK-Spark-witnessed-wall-v1",
        backend="sealed-GB10-wall-pack-v1",
        adapter_sha256=l.ADAPTER,
        executable_products_sha256=dict(l.PRODUCTS),
        parent_sha256=A,
        parent_inputs_sha256={"manifest.json": A, "material.grid": F},
    )
    return old, new


@pytest.mark.parametrize(
    "key",
    [
        *l.META,
        "parent_sha256",
        "parent_inputs_sha256",
        "backend",
        "producer_schema",
        "adapter_sha256",
        "executable_products_sha256",
        "format",
    ],
)
def test_wall_strict(key):
    a, b = edges()
    assert l.wall(a, b, B, C) == A
    b[key] = [True] if key in l.META else "unqualified"
    with pytest.raises(ValueError):
        l.wall(a, b, B, C)


def test_wall_wrong_pin():
    with pytest.raises(ValueError):
        l.wall(*edges(), A, A)


@pytest.mark.parametrize(
    "field",
    ["backend", "qualification_sha256", "pair", "anchors", "edge_manifest_sha256", "certificate"],
)
def test_parcel(field):
    a = dict(
        edge_manifest_sha256=A,
        pair=[0, 1],
        anchors=[[0, 0], [1, 1]],
        certificate={"cells": 12},
        backend={},
        qualification_sha256=E,
    )
    b = dict(
        a, edge_manifest_sha256=B, backend={"gpu": "NVIDIA GB10"}, qualification_sha256=l.LABEL_Q
    )
    l.parcel(a, b, C, D, A, B, [0, 1])
    b[field] = {"gpu": "x"} if field == "backend" else None
    with pytest.raises(ValueError):
        l.parcel(a, b, C, D, A, B, [0, 1])


@pytest.mark.parametrize(
    "field", ["parcel_manifest_sha256", "edge_manifest_sha256", "shape_ha", "summary"]
)
def test_interface(field):
    a = dict(
        format="f",
        protocol_sha256=E,
        measurement=dict(
            parcel_manifest_sha256=C,
            edge_manifest_sha256=A,
            shape_ha=[1, 3],
            summary={"cells": 3},
        ),
    )
    b = dict(
        a,
        measurement=dict(a["measurement"], parcel_manifest_sha256=D, edge_manifest_sha256=B),
        device={"gpu": "NVIDIA GB10", "redzones_verified": True},
        qualification_sha256=l.FACE_Q,
    )
    l.face(a, b, E, F, A, B, C, D)
    b["measurement"][field] = None
    with pytest.raises(ValueError):
        l.face(a, b, E, F, A, B, C, D)


@pytest.mark.parametrize(
    "action", ["block", "swap", "last", "shape", "parent", "protocol", "type", "mask", "labels"]
)
def test_roi_last_partial(action):
    m = bytes(range(15))
    v = bytes(range(60))
    x = {
        "format": "CFDK-periodic-row-index-v1",
        "protocol_sha256": "a05630e04c348abf03897082c74740d7ac00a4ad3246ab7094873089f58530a0",
        "parcel_manifest_sha256": A,
        "shape_ha": [5, 3],
        "block_rows": 4,
        "blocks": [[l.sha(m[:12]), l.sha(v[:48])], [l.sha(m[12:]), l.sha(v[48:])]],
    }
    l.index_blocks(x, A, [5, 3], m, v)
    if action == "block":
        x["block_rows"] = 3
    if action == "swap":
        x["blocks"].reverse()
    if action == "last":
        x["blocks"].pop()
    if action == "shape":
        x["shape_ha"] = [3, 5]
    if action == "parent":
        x["parcel_manifest_sha256"] = B
    if action == "protocol":
        x["protocol_sha256"] = C
    if action == "type":
        x["block_rows"] = 4.0
    if action == "mask":
        m = m[:-1]
    if action == "labels":
        v = v[:-1]
    with pytest.raises(ValueError):
        l.index_blocks(x, A, [5, 3], m, v)
