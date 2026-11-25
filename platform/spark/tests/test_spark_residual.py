"""Independent scalar and native GB10 checks of signed moments, tie order, bands and provenance."""

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_residual as br  # noqa: E402
import qualify_residual as qr  # noqa: E402
import residual_witness as rw  # noqa: E402


@pytest.fixture(scope="module")
def exe(tmp_path_factory):
    # A fresh two-build gate, not a developer's unchecked prior executable.
    with tempfile.TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as tmp:
        out = Path(tmp) / "accepted"
        b = br.build(out)
        assert (
            br.sha(out / "build.json") == qr.BUILD
            and b["independent_elf_sha256"] == [br.sha(out / "volume-residual")] * 2
        )
        yield out / "volume-residual"


def native(exe, tmp, shape, a, b, band=7, budget=8):
    aa = tmp / "candidate.f32"
    bb = tmp / "reference.f32"
    aa.write_bytes(a)
    bb.write_bytes(b)
    return subprocess.run(
        list(map(str, [exe, aa, bb, *shape, band, budget])), capture_output=True, timeout=20
    )


def f32(vals):
    return np.asarray(vals, dtype="<f4").tobytes()


@pytest.mark.parametrize(
    "kind", ["zeros", "thresholds", "extremes", "subnormals", "cancellation"]
)
def test_native_analytic(exe, tmp_path, kind):
    if kind == "zeros":
        a = f32([-0.0, 0.0, -0.0])
        b = f32([0.0, -0.0, 0.0])
        special = (3, [0] * 4, 0.0, 0)
    if kind == "thresholds":
        values = [
            q
            for c in rw.CUTS
            for q in (c, -c, float(np.nextafter(np.float32(c), np.float32(np.inf))))
        ]
        a = f32(values)
        b = f32([0.0] * len(values))
        special = (0, [10, 7, 4, 1], values[-1], 11)
    if kind == "subnormals":
        u = 2.0**-149
        a = f32([u, -u, 0])
        b = f32([-u, u, 0])
        special = (1, [0] * 4, 2 * u, 0)
    if kind == "extremes":
        mx = float(np.finfo(np.float32).max)
        a = f32([mx, -mx, mx, -mx])
        b = f32([-mx, mx, mx, -mx])
        special = (2, [2] * 4, 2 * mx, 0)
    if kind == "cancellation":
        a = f32([2.0**60, 1.0, -(2.0**60), 8.0])
        b = f32([0.0] * 4)
        special = (0, [4] * 4, 2.0**60, 0)
    shape = [len(a) // 4, 1, 1]
    r = native(exe, tmp_path, shape, a, b)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    qr.accounts(d, shape, 7, 8)
    row = d["rows"][0]
    assert (row[1], row[2:6], row[6], row[7]) == special
    rw.agree(row, rw.scalar(a, b))
    rw.agree(row, qr.numpy_oracle(a, b))
    s = qr.summarize([row], shape[0], shape[1])
    if kind in ("zeros", "thresholds", "cancellation"):
        assert s["rel_l2"] is None


def test_odd_volume_three_band_boundaries_and_tied_first(exe, tmp_path):
    nx, ny, nz = 37, 43, 15
    n = nx * ny
    shape = [nx, ny, nz]
    raw = np.arange(n * nz).reshape(nz, n).astype("<f4") / 8192
    other = raw.copy()
    raw[8, 1024] += 2
    raw[8, 1023] -= 2
    raw[10, 160] += 2
    recs = []
    for band in (1, 7, 64):
        t = native(exe, tmp_path, shape, raw.tobytes(), other.tobytes(), band)
        assert t.returncode == 0, t.stderr
        d = json.loads(t.stdout)
        qr.accounts(d, shape, band, 8)
        recs.append(d["rows"])
    assert recs[0] == recs[1] == recs[2]
    for i, row in enumerate(recs[0]):
        rw.agree(row, rw.scalar(raw[i].tobytes(), other[i].tobytes()))
    s = qr.summarize(recs[0], nx, ny)
    assert (
        s["first_maximum_xyz"] == [1023 % nx, 1023 // nx, 8]
        and s["strict_exceedance"] == [3] * 4
        and s["numeric_equal"] == n * nz - 3
    )


@pytest.mark.parametrize(
    "kind",
    [
        "nan-candidate",
        "inf-reference",
        "minus-inf",
        "short-first",
        "extra-last",
        "bad-size",
        "budget",
        "zero-band",
        "negative-width",
        "numeric-junk",
    ],
)
def test_no_stdout_or_partial_report_on_bad_volume(exe, tmp_path, kind):
    a = b = f32([1.0] * 6)
    shape = [2, 1, 3]
    band = 1
    budget = 1
    if kind == "nan-candidate":
        a = a[:-4] + struct.pack("<f", float("nan"))
    if kind == "inf-reference":
        b = b[:-4] + struct.pack("<f", float("inf"))
    if kind == "minus-inf":
        a = a[:-4] + struct.pack("<f", float("-inf"))
    if kind == "short-first":
        a = a[:-1]
    if kind == "extra-last":
        b += b"\0"
    if kind == "bad-size":
        shape = [2048] * 3
    if kind == "budget":
        shape = [512, 512, 1]
    if kind == "zero-band":
        band = 0
    if kind == "negative-width":
        shape[0] = "-1"
    if kind == "numeric-junk":
        budget = "1e1"
    r = native(exe, tmp_path, shape, a, b, band, budget)
    assert r.returncode == 2 and not r.stdout and b"volume-residual:" in r.stderr
    if kind in ("budget", "bad-size"):
        assert b"budget" in r.stderr
    if kind in ("nan-candidate", "inf-reference", "minus-inf"):
        assert b"nonfinite" in r.stderr


def doc_and_slice(tmp):
    a = f32([3, -1, 0, 4])
    b = f32([1, 1, 0, 2])
    row = rw.scalar(a, b)
    doc = dict(
        format="CFDK-paired-volume-qualification-v1",
        protocol_sha256=rw.RULES,
        row_fields=rw.FIELDS,
        shape_xyz=[2, 2, 1],
        planes=[dict(z=0, candidate_sha256=rw.hashed(a), reference_sha256=rw.hashed(b), gpu=row)],
    )
    ca = tmp / "a"
    re = tmp / "b"
    ca.write_bytes(a)
    re.write_bytes(b)
    return doc, ca, re


@pytest.mark.parametrize(
    "kind",
    [
        "valid",
        "source-swap",
        "changed-candidate",
        "truncated",
        "changed-root",
        "wrong-z",
        "false-first",
        "false-count",
        "false-energy",
        "false-bias",
        "protocol",
        "wrong-axes",
        "wrong-slice-index",
    ],
)
def test_offline_selected_slice(tmp_path, kind):
    doc, a, b = doc_and_slice(tmp_path)
    z = 0
    row = doc["planes"][0]
    if kind == "source-swap":
        a, b = b, a
    if kind == "changed-candidate":
        a.write_bytes(b"x" * 16)
    if kind == "truncated":
        b.write_bytes(b.read_bytes()[:-1])
    if kind == "wrong-z":
        z = 1
    if kind == "false-first":
        row["gpu"][7] = 1
    if kind == "false-count":
        row["gpu"][2] = 0
    if kind == "false-energy":
        row["gpu"][11] = 0.0
    if kind == "false-bias":
        row["gpu"][8] += 1.0
    if kind == "protocol":
        doc["protocol_sha256"] = "0" * 64
    if kind == "wrong-axes":
        doc["row_fields"] = list(reversed(rw.FIELDS))
    if kind == "wrong-slice-index":
        row["z"] = 1
    report = tmp_path / "qualification.json"
    report.write_text(json.dumps(doc))
    pin = br.sha(report) if kind != "changed-root" else "0" * 64
    if kind == "valid":
        result = rw.verify(report, pin, z, a, b)
        assert (
            result["maximum_xyz"] == [0, 0, 0]
            and result["max_abs"] == 2
            and result["strict_exceedance"] == [3] * 4
        )
    else:
        with pytest.raises(ValueError):
            rw.verify(report, pin, z, a, b)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown",
        "extra-pine-pin-on-walnut",
        "missing-pine-qualification",
        "damaged-walnut-parent",
    ],
)
def test_never_substitute_another_measured_parent(tmp_path, kind):
    if kind == "unknown":
        with pytest.raises(ValueError):
            qr.volumes("fake", tmp_path)
    if kind == "extra-pine-pin-on-walnut":
        with pytest.raises(ValueError):
            qr.volumes("walnut720", tmp_path, "1" * 64)
    if kind == "missing-pine-qualification":
        with pytest.raises(FileNotFoundError):
            qr.volumes("pine720", tmp_path, "1" * 64)
    if kind == "damaged-walnut-parent":
        folder = tmp_path / "walnut-astra-gb10-001"
        folder.mkdir()
        (folder / "result.json").write_text("{}")
        with pytest.raises(ValueError):
            qr.volumes("walnut720", tmp_path)
