"""GB10 mapped vs staged differential tests, including fail-before-publication cases."""

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mapped_transport as t
import spark_residual_audit as audit
from qualify_mapped_guards import cases
from qualify_residual import numpy_oracle
from residual_witness import agree

M = t.ROOT.parent
EXE = M / "mapped-residual-build-001/mapped-residual"
COPY = M / "residual-builds-001/qualified/volume-residual"


def launch(exe, a, b, shape, band, budget):
    return subprocess.run(
        list(map(str, [exe, a, b, *shape, band, budget])), capture_output=True, timeout=60
    )


@pytest.mark.parametrize("band", [1, 4, 16, 64])
def test_real_partial_ties_and_oracle(tmp_path, band):
    shape, a, b = cases()
    a[0, 0], b[0, 0] = -0.0, +0.0
    a[-1] = b[-1] = 0.0
    a[-1, [1023, 1024, -1]] = [2, -2, 2]
    pa, pb = tmp_path / "candidate", tmp_path / "reference"
    a.tofile(pa)
    b.tofile(pb)
    before = [t.sha(p) for p in (pa, pb)]
    ds = []
    for exe in (COPY, EXE):
        p = launch(exe, pa, pb, shape, band, 1)
        assert p.returncode == 0 and p.stderr == b""
        d = json.loads(p.stdout)
        ds.append(d)
    t.mapped_account(ds[1], shape, band, 1)
    assert t.signature(ds[0]["rows"]) == t.signature(ds[1]["rows"])
    for i, row in enumerate(ds[1]["rows"]):
        agree(row, numpy_oracle(a[i].tobytes(), b[i].tobytes()))
    assert ds[1]["rows"][0][7] == 0 and ds[1]["rows"][-1][7] == 1023
    assert [t.sha(p) for p in (pa, pb)] == before


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "long",
        "nan",
        "+inf",
        "-inf",
        "band0",
        "budget0",
        "budget_negative",
        "grid0",
        "grid_large",
        "budget_small",
    ],
)
def test_native_refuses_before_json(tmp_path, fault):
    s, a, b = cases()
    pa = tmp_path / "candidate"
    pb = tmp_path / "reference"
    band, cap = 4, 1
    if fault in ("nan", "+inf", "-inf"):
        b[-1, -1] = np.float32(float(fault))
    a.tofile(pa)
    b.tofile(pb)
    if fault == "short":
        pb.write_bytes(pb.read_bytes()[:-1])
    if fault == "long":
        pa.write_bytes(pa.read_bytes() + b"x")
    if fault == "band0":
        band = 0
    if fault == "budget0":
        cap = 0
    if fault == "budget_negative":
        cap = -1
    if fault == "grid0":
        s[2] = 0
    if fault == "grid_large":
        s[1] = 2049
    if fault == "budget_small":
        s = [512, 512, 512]
        pa = tmp_path / "never_open_this"
    files = {p.name: t.sha(p) for p in tmp_path.iterdir()}
    p = launch(EXE, pa, pb, s, band, cap)
    assert p.returncode != 0 and not p.stdout and p.stderr
    if fault == "budget_small":
        assert b"budget exceeded" in p.stderr
    assert files == {p.name: t.sha(p) for p in tmp_path.iterdir()}


@pytest.fixture
def profile():
    return t.json_checked(M / "mapped-residual-bench-001/pairs.json", audit.PROFILE)


@pytest.mark.parametrize(
    "cap,band,policy,result",
    [
        (9, None, "latency", (4, "mapped")),
        (8, None, "latency", None),
        (96, 16, "latency", (16, "copy")),
        (33, 16, "memory", (16, "mapped")),
        (34, 16, "latency", (16, "mapped")),
        (65, 16, "latency", (16, "copy")),
        (16, 4, "latency", (4, "mapped")),
        (96, None, "memory", (4, "mapped")),
        (0, 4, "memory", None),
        (True, 4, "latency", None),
    ],
)
def test_selector(profile, cap, band, policy, result):
    f = lambda: audit.choose(profile["summary"], profile["shape"], cap, band, policy)
    if result is None:
        with pytest.raises(ValueError):
            f()
    else:
        assert f() == result


@pytest.mark.parametrize(
    "key",
    [
        "order",
        "rows_sha256",
        "combined_work_bytes",
        "h2d_bytes",
        "d2h_bytes",
        "mode",
        "child_elapsed_ms",
    ],
)
def test_bad_profile(profile, key):
    rows = profile["samples"]
    if key == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif key == "child_elapsed_ms":
        rows[-1][key] = math.nan
    else:
        rows[-1][key] = "different"
    with pytest.raises(ValueError):
        t.decision(rows, profile["shape"], profile["rows_sha256"])


@pytest.mark.parametrize(
    "time,wins,prefer", [(95.0, 5, "mapped"), (95.01, 6, "copy"), (94.0, 4, "copy")]
)
def test_predeclared_gate(profile, time, wins, prefer):
    for r in profile["samples"]:
        r["child_elapsed_ms"] = (
            100.0 if r["mode"] == "copy" else (time if r["block"] < wins else 101.0)
        )
    table = t.decision(profile["samples"], profile["shape"], profile["rows_sha256"])
    assert all(x["latency_choice"] == prefer for x in table.values())


def test_abi_signed_zero_and_types():
    row = [0] * 12
    h = t.signature([row])
    row[8] = -0.0
    assert h != t.signature([row])
    row[8] = float("nan")
    with pytest.raises(ValueError):
        t.signature([row])
    row[8] = 0
    row[1] = True
    with pytest.raises(ValueError):
        t.signature([row])


@pytest.mark.parametrize(
    "k,v",
    [
        ("can_map_host", 0),
        ("uva", True),
        ("gpu", "other"),
        ("guard_checks", 4),
        ("h2d_bytes", 4),
        ("d2h_bytes", 1),
        ("mapped_bytes", 0),
        ("read_ms", float("inf")),
    ],
)
def test_capability_and_counters(k, v):
    d = json.loads((M / "mapped-residual-audit-001/native.json").read_bytes())
    d[k] = v
    with pytest.raises(ValueError):
        t.mapped_account(d, [512] * 3, 4, 9)


def test_cli_fails_without_output(tmp_path):
    with pytest.raises(ValueError):
        with t.held("walnut720", M / "residual-walnut-001/qualification.json", "0" * 64):
            pytest.fail("opened wrong parent")
    dest = tmp_path / "unpublished"
    p = subprocess.run(
        [
            sys.executable,
            "-B",
            str(HERE / "spark_residual_audit.py"),
            str(dest),
            "--cap-mib",
            "8",
        ],
        capture_output=True,
        timeout=30,
    )
    assert (
        p.returncode
        and not p.stdout
        and b"no qualified configuration" in p.stderr
        and not dest.exists()
    )
