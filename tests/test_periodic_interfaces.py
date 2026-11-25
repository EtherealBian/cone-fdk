"""Analytic face totals and altered-evidence controls. Device trials are run separately."""

import hashlib
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import periodic_interfaces as pi
import pytest
from build_periodic_labels import sha
from interface_ledger import MAX_COMPONENTS, capacity, ledger
from qualify_periodic_interfaces import BUILD_NAMES, PROTOCOL, account, direct, vector
from qualify_periodic_labels import flood


def test_all_ternary_faces_and_lut():
    for t in itertools.product(range(3), repeat=6):
        m = np.array(t, "u1").reshape(2, 3)
        np.testing.assert_array_equal(direct(m), vector(m))
    # N is exterior, E below, S unknown, W measured across the seam.
    m = np.array([[1, 0, 0, 1], [2, 0, 0, 0]], "u1")
    assert direct(m)[0, 0] == 3 + (1 << 2) + (2 << 4)


def inputs(tmp, m):
    a, b, c = (tmp / n for n in ("mask", "labels", "faces"))
    m = np.asarray(m, "u1")
    labels = flood(m)
    m.tofile(a)
    labels.tofile(b)
    direct(m).tofile(c)
    exp = dict(
        mask_u8_sha256=sha(a),
        labels_i32le_sha256=sha(b),
        observed_components=int(np.unique(labels[labels > 0]).size),
        observed_flagged_cells=int(np.sum(m == 1)),
    )
    return a, b, c, list(m.shape), exp


@pytest.mark.parametrize("band", [1, 2, 7])
def test_ring_has_two_measured_circumferences_no_seam_cut(tmp_path, band):
    m = np.zeros((3, 5), "u1")
    m[1] = 1
    q = ledger(*inputs(tmp_path, m), band)
    assert q["components"] == [
        dict(
            id=6,
            area_cells=5,
            internal_undirected_edges=5,
            faces=[[5, 0, 0], [0, 0, 0], [5, 0, 0], [0, 0, 0]],
        )
    ]


@pytest.mark.parametrize("v,k", [(0, 0), (2, 1)])
def test_hole_kind_and_axial_clipping_are_distinct(tmp_path, v, k):
    m = np.ones((3, 5), "u1")
    m[1, 2] = v
    q = ledger(*inputs(tmp_path, m), 2)
    f = np.array(q["faces_by_direction_and_kind"])
    np.testing.assert_array_equal(f[:, k], [1, 1, 1, 1])
    np.testing.assert_array_equal(f[:, 1 - k], [0] * 4)
    np.testing.assert_array_equal(f[:, 2], [5, 0, 5, 0])
    assert len(q["components"]) == 1


def test_one_row_records_two_cut_faces_per_measured_cell(tmp_path):
    q = ledger(*inputs(tmp_path, [[1, 1, 1, 1, 1]]), 1)
    assert q["components"] == [
        dict(
            id=1,
            area_cells=5,
            internal_undirected_edges=5,
            faces=[[0, 0, 5], [0, 0, 0], [0, 0, 5], [0, 0, 0]],
        )
    ]


def test_diagonal_islands_keep_individual_faces(tmp_path):
    q = ledger(*inputs(tmp_path, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]), 1)
    assert [c["id"] for c in q["components"]] == [1, 6] and all(
        c["internal_undirected_edges"] == 0 for c in q["components"]
    )
    assert q["faces_by_direction_and_kind"] == [[1, 0, 1], [2, 0, 0], [2, 0, 0], [2, 0, 0]]


@pytest.mark.parametrize(
    "bad",
    [
        "face",
        "trailing",
        "truncated",
        "different_mask",
        "different_labels",
        "census",
        "false_join",
    ],
)
def test_weighted_totals_cannot_replace_exact_face_comparison(tmp_path, bad):
    a, b, c, shape, exp = inputs(tmp_path, [[1, 1, 0, 1], [1, 0, 2, 0], [0, 1, 0, 1]])
    if bad == "face":
        v = bytearray(c.read_bytes())
        v[0] ^= 1
        c.write_bytes(v)
    elif bad == "trailing":
        c.write_bytes(c.read_bytes() + b"\0")
    elif bad == "truncated":
        b.write_bytes(b.read_bytes()[:-4])
    elif bad == "different_mask":
        exp["mask_u8_sha256"] = "0" * 64
    elif bad == "different_labels":
        exp["labels_i32le_sha256"] = "f" * 64
    elif bad == "census":
        exp["observed_components"] -= 1
    elif bad == "false_join":
        data = np.fromfile(b, "<i4")
        data[1] = 2
        data.tofile(b)
        exp["labels_i32le_sha256"] = sha(b)
    with pytest.raises(ValueError):
        ledger(a, b, c, shape, exp, 1)


@pytest.mark.parametrize("n", [True, -1, MAX_COMPONENTS + 1])
def test_no_unbounded_component_dictionary(n):
    with pytest.raises(ValueError, match="supported count"):
        capacity({"observed_components": n})


def prepared(tmp, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    b = tmp / "source"
    tool = tmp / "build"
    tool.mkdir()
    pp = "961648e72ce25d1e3d36b1f8a576af3e63750a2b809ca4e160a183e075ba2395"
    z = root / "reports/periodic-seam-demo.zip"
    assert sha(z) == "d8c833a2de5e3dea4101fa9605f95834bd8ea7d1618ff5d421ea485be7142078"
    with zipfile.ZipFile(z) as f:
        f.extractall(b)
    for n in BUILD_NAMES:
        (tool / n).write_bytes(b"test fixture")
    qualification = tmp / "qualified.json"
    qualification.write_text(
        json.dumps(
            dict(
                format="CFDK-periodic-interface-qualification-v1",
                protocol_sha256=PROTOCOL,
                build=dict(products={n: sha(tool / n) for n in BUILD_NAMES}, identical_builds=2),
            )
        )
    )

    def stub(build, src, dst, h, w, maps, band, budget):
        m = np.fromfile(src, "u1").reshape(h, w)
        vector(m).tofile(dst)
        k = (h + band - 1) // band
        c = min(h, band)
        d = dict(
            format="CFDK-periodic-interfaces-v1",
            shape_ha=[h, w],
            maps=1,
            band_rows=band,
            read_bytes=h * w,
            d2h_bytes=h * w,
            kernel_launches=k,
            h2d_bytes=h * w + 2 * w * k,
            peak_owned_bytes=4 * ((w * (c + 2) + 3) // 4 + (w * c + 3) // 4) + 1024,
            budget_bytes=budget << 20,
            redzones_verified=True,
            gpu="fixture",
            driver=0,
        )
        account(d, h, w, 1, band, budget)
        return d

    monkeypatch.setattr(pi, "native", stub)
    return b, tmp / "result", tool, qualification, pp, sha(qualification), 7


def test_component_ledger_portable_and_band_neutral(tmp_path, monkeypatch):
    b, out, tool, q, pp, qp, band = prepared(tmp_path, monkeypatch)
    d = pi.generate(b, out, tool, q, pp, qp, band)
    r = pi.verify(b, out, pp, sha(out / "manifest.json"), 1)
    assert r == d["measurement"]
    assert r["summary"]["components"][0]["area_cells"] == 642
    assert len(r["summary"]["components"]) == 1
    assert not any(x[1] for x in r["summary"]["faces_by_direction_and_kind"])


@pytest.mark.parametrize("bad", ["source", "binary", "qualification", "occupied"])
def test_no_publish_from_invalid_evidence(tmp_path, monkeypatch, bad):
    b, out, tool, q, pp, qp, band = prepared(tmp_path, monkeypatch)
    if bad == "source":
        (b / "labels.i32").write_bytes(b"bad")
    elif bad == "binary":
        (tool / "periodic-interfaces.exe").write_bytes(b"changed")
    elif bad == "qualification":
        qp = "0" * 64
    elif bad == "occupied":
        out.mkdir()
        (out / "keep").write_bytes(b"prior")
    monkeypatch.setattr(pi, "native", lambda *a: pytest.fail("GPU should not be invoked"))
    with pytest.raises(ValueError):
        pi.generate(b, out, tool, q, pp, qp, band)
    if bad == "occupied":
        assert (out / "keep").read_bytes() == b"prior"
    else:
        assert not out.exists()


def test_device_result_is_not_trusted_on_its_own(tmp_path, monkeypatch):
    args = prepared(tmp_path, monkeypatch)
    prior = pi.native

    def corrupt(*a):
        result = prior(*a)
        dst = a[2]
        data = bytearray(dst.read_bytes())
        data[-1] ^= 1
        dst.write_bytes(data)
        return result

    monkeypatch.setattr(pi, "native", corrupt)
    with pytest.raises(ValueError, match="interface byte"):
        pi.generate(*args)
    assert not args[1].exists()


def test_precommitted_protocol():
    root = Path(__file__).resolve().parents[1]
    assert (
        hashlib.sha256(
            (root / "fixtures/periodic-interface-protocol-v1.json").read_bytes()
        ).hexdigest()
        == PROTOCOL
    )
