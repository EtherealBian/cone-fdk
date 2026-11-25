"""Parcel tests use an explicitly mocked device only; hardware qualification is a separate gate."""

import json
from pathlib import Path

import numpy as np
import periodic_parcel as pp
import pytest
from build_periodic_labels import sha
from periodic_label_cpu import cpu
from test_stripe_batch import recipe_at


def cmd(tmp, monkeypatch):
    _, _, recipe = recipe_at(tmp)
    e = recipe["entries"][0]

    def standin(build, qualification, qual_pin, m, lab, h, w, budget):
        cpu(m, lab, h, w, 1)
        return {"test_only": True}

    monkeypatch.setattr(pp, "native", standin)
    return dict(
        job=tmp / "edge",
        dest=tmp / "result",
        build=tmp / "tools",
        qualification=tmp / "trusted",
        qual_pin="0" * 64,
        edge_pin=e["manifest_sha256"],
        pair=[1, 1],
        anchors=e["anchors_row_col"],
        band=2,
        budget=32,
    )


def test_validated_parcel_is_portable_and_independent_of_band(tmp_path, monkeypatch):
    args = cmd(tmp_path, monkeypatch)
    d = pp.generate(**args)
    bundle = args["dest"]
    expected = sha(bundle / "manifest.json")
    for band in (1, 3, 32):
        r = pp.offline(bundle, expected, band)
        assert r["source_scenario"] == d["source_scenario"]
        assert r["certificate"]["labels_i32le_sha256"] == sha(bundle / "labels.i32")
    assert not any("partial" in p.name for p in tmp_path.iterdir())


@pytest.mark.parametrize(
    "mode",
    [
        "wrong-parent",
        "pair-outside",
        "bool-pair",
        "bool-budget",
        "source-overlap",
        "existing-output",
        "unselected-wall",
        "saved-envelope",
        "mislabel",
        "gpu-failure",
        "race",
    ],
)
def test_never_publish_partial_or_overwrite(tmp_path, monkeypatch, mode):
    a = cmd(tmp_path, monkeypatch)
    dst = a["dest"]
    if mode == "wrong-parent":
        a["edge_pin"] = "f" * 64
    elif mode == "pair-outside":
        a["pair"] = [0, 3]
    elif mode == "bool-pair":
        a["pair"] = [True, 1]
    elif mode == "bool-budget":
        a["budget"] = True
    elif mode == "source-overlap":
        a["dest"] = a["job"] / "inside"
    elif mode == "existing-output":
        dst.mkdir()
        (dst / "keep").write_bytes(b"keep")
    elif mode in ("unselected-wall", "saved-envelope"):
        p = a["job"] / ("inspected-02.wall" if mode == "unselected-wall" else "class.u8")
        b = p.read_bytes()
        p.write_bytes(b[:-1] + bytes([b[-1] ^ 1]))
    else:
        old = pp.native

        def altered(*params):
            if mode == "gpu-failure":
                raise RuntimeError("device unavailable")
            d = old(*params)
            if mode == "race":
                dst.mkdir()
                (dst / "keep").write_bytes(b"keep")
            else:
                f = params[4]
                data = np.fromfile(f, "<i4")
                data[:] = 0
                data.tofile(f)
            return d

        monkeypatch.setattr(pp, "native", altered)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        pp.generate(**a)
    if mode in ("existing-output", "race"):
        assert [p.name for p in dst.iterdir()] == ["keep"] and (
            dst / "keep"
        ).read_bytes() == b"keep"
    else:
        assert not a["dest"].exists()
    assert not list(tmp_path.glob(".periodic-*"))


@pytest.mark.parametrize(
    "mode",
    [
        "wrong-pin",
        "mask",
        "labels",
        "manifest",
        "census",
        "anchors",
        "levels",
        "pair",
        "work",
        "band",
    ],
)
def test_offline_rejects_damage_or_contradictory_metadata(tmp_path, monkeypatch, mode):
    a = cmd(tmp_path, monkeypatch)
    d = pp.generate(**a)
    dst = a["dest"]
    mf = dst / "manifest.json"
    digest = sha(mf)
    if mode == "wrong-pin":
        digest = "f" * 64
    elif mode in ("mask", "labels"):
        p = dst / ("mask.u8" if mode == "mask" else "labels.i32")
        p.write_bytes(p.read_bytes()[:-1])
    elif mode == "manifest":
        mf.write_bytes(mf.read_bytes() + b" ")
    else:
        if mode == "census":
            d["source_scenario"]["unknown_cells"] += 1
        elif mode == "anchors":
            d["anchors_row_col"][0] = [-1, 0]
        elif mode == "levels":
            d["levels_per_mm"].reverse()
        elif mode == "pair":
            d["pair"][0] = 1.0
        elif mode == "work":
            d["certificate"]["peak_band_cells"] += 1
        else:
            d["producer_band_rows"] = True
        mf.write_text(json.dumps(d))
        digest = sha(mf)
    with pytest.raises((ValueError, OSError)):
        pp.offline(dst, digest, 1)


def test_untrusted_exe_is_not_launched(tmp_path, monkeypatch):
    import qualify_periodic_labels as q

    exe, ptx = (tmp_path / n for n in ("periodic-labels.exe", "periodic-labels.ptx"))
    exe.write_bytes(b"x")
    ptx.write_bytes(b"y")
    record = {
        "format": "CFDK-periodic-label-qualification-v1",
        "protocol_sha256": q.PROTOCOL,
        "build": {"identical_builds": 2, "products": {exe.name: sha(exe), ptx.name: sha(ptx)}},
    }
    qualification = tmp_path / "qual"
    qualification.write_text(json.dumps(record))
    exe.write_bytes(b"z")
    monkeypatch.setattr(q, "runproc", lambda *a: pytest.fail("must reject before running"))
    with pytest.raises(ValueError, match="backend bytes"):
        pp.native(
            tmp_path, qualification, sha(qualification), Path("unused"), Path("unused"), 2, 3, 32
        )
