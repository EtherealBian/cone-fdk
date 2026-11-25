"""Analytic cylinder, fully re-sealed counterexamples, malformed transport and recovery."""

import io
import json
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import offline_campaign as oc  # noqa: E402
import pack_campaign as pc  # noqa: E402
import resume_campaign as rc  # noqa: E402


def example():
    mask = bytes([1, 0, 2, 1])
    labels = struct.pack("<4i", 1, 0, -1, 1)
    faces = bytes([55, 0, 0, 179])
    p0 = [0, 0]
    ha = [1, 4]
    scen = dict(
        pair=p0,
        observed_components=1,
        observed_flagged_cells=2,
        largest_observed_cells=2,
        unknown_cells=1,
        anchor_cells=["flagged", "flagged"],
        anchor_state="observed_link",
    )
    cert = dict(
        observed_components=1, observed_flagged_cells=2, unknown_cells=1, verified_cells=4
    )
    p = dict(
        format="CFDK-periodic-label-parcel-v1",
        shape_ha=ha,
        pair=p0,
        qualification_sha256="2" * 64,
        edge_manifest_sha256="3" * 64,
        certificate=cert,
        source_scenario=scen,
        anchors_row_col=[[0, 0], [0, 3]],
    )
    table = [[0, 0, 2], [1, 0, 0], [0, 0, 2], [0, 1, 0]]
    s = dict(
        directions=["north", "east", "south", "west"],
        face_kinds=["measured_below", "unknown", "axial_cut"],
        components=[dict(id=1, area_cells=2, internal_undirected_edges=1, faces=table)],
        faces_by_direction_and_kind=table,
        observed_cells=2,
        source_cells=4,
    )
    f = dict(
        format="CFDK-certified-interface-ledger-v1",
        protocol_sha256=oc.FACE,
        qualification_sha256="4" * 64,
        measurement=dict(edge_manifest_sha256="3" * 64, pair=p0, shape_ha=ha, summary=s),
    )
    x = dict(
        format="CFDK-periodic-row-index-v1", protocol_sha256=oc.ROI, shape_ha=ha, block_rows=1
    )
    return p, f, x, mask, labels, faces


def wire(fields=None, mutate=None):
    # Independent test encoder. Re-pin deliberately bad measurement bytes: a hash-only
    # implementation would accept these jobs but the semantic verifier must not.
    p, f, x, m, dense, g = fields or example()
    s = f["measurement"]["summary"]
    p["certificate"].update(mask_u8_sha256=oc.sha(m), labels_i32le_sha256=oc.sha(dense))
    s["face_u8_sha256"] = oc.sha(g)
    pb = oc.encoded(p)
    pp = oc.sha(pb)
    f["measurement"]["parcel_manifest_sha256"] = x["parcel_manifest_sha256"] = pp
    x["blocks"] = [[oc.sha(m), oc.sha(dense)]]
    if mutate:
        mutate("docs", p, f, x)
    data = [pb, oc.encoded(f), oc.encoded(x), m, dense, g]
    row = dict(
        case="analytic",
        pair=[0, 0],
        old_manifest_sha256="9" * 64,
        **dict(zip(oc.SLOTS, map(oc.sha, data))),
    )
    survey = dict(
        format="CFDK-periodic-parcel-campaign-v1",
        recipe_sha256="1" * 64,
        qualification_sha256="2" * 64,
        maps=1,
        pair_cells=4,
        cases={
            "analytic": dict(
                status="complete_coverage",
                rows=[
                    dict(
                        pair=[0, 0],
                        manifest_sha256=pp,
                        mask_sha256=oc.sha(m),
                        labels_sha256=oc.sha(dense),
                        anchor_state=p["source_scenario"]["anchor_state"],
                    )
                ],
            ),
            "withheld": dict(status="upstream_refused", reason="missing-reference"),
        },
    )
    if mutate:
        mutate("survey", survey, None, None)
    sb = oc.encoded(survey)
    c = dict(
        format="CFDK-GB10-source-to-interface-v1",
        source_recipe_sha256="1" * 64,
        spark_survey_sha256=oc.sha(sb),
        label_qualification_sha256="2" * 64,
        interface_qualification_sha256="4" * 64,
        maps=1,
        cells=4,
        rows=[row],
        upstream_refusals=["withheld"],
        scope="analytic only",
    )
    if mutate:
        mutate("chain", c, None, None)
    cb = oc.encoded(c)
    out = bytearray(oc.MAGIC)
    seen = set()
    for datum in (cb, sb, *data):
        if oc.sha(datum) not in seen:
            out.extend(struct.pack("<I", len(datum)) + datum)
            seen.add(oc.sha(datum))
    return bytes(out), oc.sha(cb)


def test_explicit_one_row_cylinder_with_unknown_and_seam(tmp_path):
    buf, pin = wire()
    a = oc.walk(io.BytesIO(buf), pin)
    assert (
        a["cells"] == 4
        and a["rows"][0]["scenario"]["anchor_state"] == "observed_link"
        and a["refused"] == {"withheld": "missing-reference"}
    )
    p = tmp_path / "renamed.bin"
    p.write_bytes(buf)
    run = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(HERE / "offline_campaign.py"), str(p), pin],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    assert json.loads(run.stdout) == a


@pytest.mark.parametrize(
    "bad",
    [
        "label",
        "face",
        "nonternary",
        "off-support",
        "forged-count",
        "anchors",
        "wrong-anchor-state",
        "wrong-face-axes",
        "component-id",
        "unknown-census",
    ],
)
def test_resealed_false_measurement_rejected(bad):
    p, f, x, m, dense, g = example()
    s = f["measurement"]["summary"]
    if bad == "label":
        dense = struct.pack("<4i", 1, 0, -1, 4)
    if bad == "face":
        g = bytes([55, 0, 0, 51])
    if bad == "nonternary":
        m = bytes([1, 3, 2, 1])
    if bad == "off-support":
        dense = struct.pack("<4i", 1, 0, 0, 1)
    if bad == "forged-count":
        s["observed_cells"] = 3
    if bad == "anchors":
        p["anchors_row_col"] = [[True, 0], [0, 3]]
    if bad == "wrong-anchor-state":
        p["source_scenario"]["anchor_state"] = "separated_on_grid"
    if bad == "wrong-face-axes":
        s["directions"].reverse()
    if bad == "component-id":
        s["components"][0]["id"] = 4
    if bad == "unknown-census":
        p["certificate"]["unknown_cells"] = 0
    buf, pin = wire([p, f, x, m, dense, g])
    with pytest.raises(ValueError):
        oc.walk(io.BytesIO(buf), pin)


@pytest.mark.parametrize(
    "bad",
    [
        "path-case",
        "duplicate-job",
        "withheld-job",
        "hidden-refusal",
        "wrong-parent",
        "wrong-edge",
        "wrong-index",
        "version",
        "old-root",
        "wrong-pair",
        "missing-coverage",
    ],
)
def test_resealed_false_provenance_rejected(bad):
    def change(when, a, b, c):
        if when == "chain":
            if bad == "path-case":
                a["rows"][0]["case"] = "../outside"
            if bad == "duplicate-job":
                a["rows"] *= 2
                a["maps"] = 2
            if bad == "withheld-job":
                a["rows"][0]["case"] = "withheld"
            if bad == "hidden-refusal":
                a["upstream_refusals"] = []
            if bad == "old-root":
                a["rows"][0]["old_manifest_sha256"] = a["rows"][0]["new_manifest_sha256"]
            if bad == "wrong-pair":
                a["rows"][0]["pair"] = [0, True]
        if when == "survey" and bad == "missing-coverage":
            a["cases"]["analytic"]["rows"][0]["pair"] = [1, 1]
        if when == "docs":
            if bad == "wrong-parent":
                b["measurement"]["parcel_manifest_sha256"] = "0" * 64
            if bad == "wrong-edge":
                b["measurement"]["edge_manifest_sha256"] = "0" * 64
            if bad == "wrong-index":
                c["blocks"] = [["1" * 64, "1" * 64]]
            if bad == "version":
                c["protocol_sha256"] = "0" * 64

    buf, pin = wire(mutate=change)
    with pytest.raises(ValueError):
        oc.walk(io.BytesIO(buf), pin)


@pytest.mark.parametrize(
    "bad",
    [
        "short",
        "missing",
        "trailing",
        "oversize",
        "digest",
        "wrong-root",
        "magic",
        "duplicate-json",
    ],
)
def test_malformed_transport(bad):
    buf, pin = wire()
    n = len(oc.MAGIC)
    if bad == "short":
        buf = buf[:-1]
    if bad == "missing":
        buf = buf[: n + 2]
    if bad == "trailing":
        buf += b"x"
    if bad == "oversize":
        buf = buf[:n] + b"\xff" * 4 + buf[n + 4 :]
    if bad == "digest":
        buf = buf[:-1] + bytes([buf[-1] ^ 1])
    if bad == "wrong-root":
        pin = "d" * 64
    if bad == "magic":
        buf = b"?" + buf[1:]
    if bad == "duplicate-json":
        cb = b'{"rows":1,"rows":2}'
        buf = oc.MAGIC + struct.pack("<I", len(cb)) + cb
        pin = oc.sha(cb)
    with pytest.raises(ValueError):
        oc.walk(io.BytesIO(buf), pin)


@pytest.mark.parametrize(
    "bad",
    [
        "damaged-receipt",
        "damaged-input",
        "foreign-job",
        "other-pin",
        "changed-pack",
        "trailing",
        "state-symlink",
        "concurrent",
        "existing-output",
        "foreign-file",
        "changed-reviewer",
    ],
)
def test_resume_fails_closed(tmp_path, bad, monkeypatch):
    buf, pin = wire()
    pack = tmp_path / "example.cfdk"
    pack.write_bytes(buf)
    state = tmp_path / "state"
    first = tmp_path / "first"
    second = tmp_path / "second"
    rc.batch(pack, pin, state, first)
    if bad == "existing-output":
        second.write_bytes(b"keep")
    if bad == "state-symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(state, target_is_directory=True)
        state = alias
    if bad == "foreign-file":
        (state / "stray").write_bytes(b"keep")
    if bad == "other-pin":
        pin = "7" * 64
    if bad == "changed-pack":
        pack.write_bytes(buf[:-1] + bytes([buf[-1] ^ 1]))
    if bad == "trailing":
        pack.write_bytes(buf + b"x")
    if bad == "changed-reviewer":
        monkeypatch.setattr(rc, "file_sha", lambda path: "8" * 64)
    if bad in ("damaged-receipt", "damaged-input", "foreign-job"):
        with sqlite3.connect(state / "jobs.sqlite3") as db:
            if bad == "damaged-receipt":
                db.execute("update jobs set receipt=x'00'")
            if bad == "damaged-input":
                db.execute("update jobs set input='x'")
            if bad == "foreign-job":
                db.execute("insert into jobs values('withheld:0:0','x',x'00')")
    from contextlib import nullcontext

    held = rc.opened(state, True) if bad == "concurrent" else nullcontext()
    with held, pytest.raises((ValueError, OSError)):
        rc.batch(pack, pin, state, second, resume=True)
    if bad == "existing-output":
        assert second.read_bytes() == b"keep"
    else:
        assert not second.exists()


def test_no_numpy_needed_and_cache_revalidated(tmp_path):
    buf, pin = wire()
    p = tmp_path / "e"
    p.write_bytes(buf)
    state = tmp_path / "state"
    a = tmp_path / "first"
    b = tmp_path / "second"
    r = rc.batch(p, pin, state, a)
    s = rc.batch(p, pin, state, b, resume=True)
    assert (r["fresh"], r["reused"], s["fresh"], s["reused"]) == (
        1,
        0,
        0,
        1,
    ) and a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize(
    "variant", ["valid", "changed-raw", "source-link", "already-published", "inside-source"]
)
def test_pack_sources_and_publication(tmp_path, variant):
    expected, pin = wire()
    stream = oc.Reader(io.BytesIO(expected))
    root = tmp_path / "source"
    root.mkdir()

    def frame(path, expected_pin):
        data = stream.get(expected_pin, oc.MAX_DOC)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(data)
        return data

    chain = oc.obj(frame(root / "chain.json", pin))
    frame(root / "parcels/result.json", chain["spark_survey_sha256"])
    row = chain["rows"][0]
    base = root / "parcels/analytic/0-0"
    face = root / "faces/analytic/0-0"
    for slot, path in zip(
        oc.SLOTS,
        [
            base / "manifest.json",
            face / "manifest.json",
            root / "roi/analytic/0-0.json",
            base / "mask.u8",
            base / "labels.i32",
            face / "face.u8",
        ],
        strict=True,
    ):
        frame(path, row[slot])
    out = tmp_path / "output"
    raw = base / "labels.i32"
    if variant == "valid":
        pc.pack(root, out, pin)
        assert out.read_bytes() == expected
        return
    if variant == "already-published":
        out.write_bytes(b"keep")
    if variant == "inside-source":
        out = root / "extra"
    if variant == "changed-raw":
        raw.write_bytes(b"wrong")
    if variant == "source-link":
        raw.rename(tmp_path / "target")
        raw.symlink_to(tmp_path / "target")
    with pytest.raises((ValueError, OSError)):
        pc.pack(root, out, pin)
    if variant == "already-published":
        assert out.read_bytes() == b"keep"
    else:
        assert not out.exists()
    assert not list(tmp_path.glob(".campaign-*"))
