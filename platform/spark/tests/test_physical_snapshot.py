"""Published measured specimen and constructed protocol controls are deliberately separate."""

import json
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pack_physical_snapshot as pack  # noqa: E402
import plot_physical_profile as chart  # noqa: E402
import review_physical_snapshot as rev  # noqa: E402

EX = HERE.parents[1] / "examples/physical-peak-walnut"
PIN = "98311760b93bf8636479d87b2b429a881b667ba649f987d2047c19dfada60370"
h = rev.rw.hashed


def dump(v):
    return json.dumps(v, sort_keys=True, allow_nan=False).encode()


def parts():
    raw = (EX / "planes.cfdk").read_bytes()
    p = len(rev.MAGIC)
    out = []
    while p < len(raw):
        n = struct.unpack_from("<I", raw, p)[0]
        out.append(raw[p + 4 : p + 4 + n])
        p += n + 4
    assert p == len(raw) and len(out) == 7
    return out


def frame(seq):
    return rev.MAGIC + b"".join(struct.pack("<I", len(v)) + v for v in seq)


def catalog():
    return json.loads((EX / "catalog.json").read_bytes())


def bind(cat, seq):
    data = frame(seq)
    cat["pack_sha256"] = h(data)
    return rev.decode(cat, data)


def test_real_public_isolated_python():
    p = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(HERE / "review_physical_snapshot.py"),
            str(EX),
            PIN,
        ],
        capture_output=True,
        check=True,
    )
    d = json.loads(p.stdout)
    w = d["cases"][0]["evidence"]
    assert (
        d["committed_cases"],
        d["planned_cases"],
        d["specimens_with_evidence"],
        d["planned_specimens"],
    ) == (1, 3, 1, 2)
    assert all("evidence" not in c for c in d["cases"][1:])
    assert (
        w["first_maximum_xyz"],
        w["plane_rows_reaggregated"],
        w["peak_plane_voxels_remeasured"],
    ) == ([367, 318, 256], 512, 262144)
    assert w["max_abs"] == 4.7359615564346313e-5 and w["peak_plane_strict_exceedance"] == [
        224588,
        125134,
        4241,
        0,
    ]


@pytest.mark.parametrize("arg,rc", [(("--require-complete",), 3), ((), 1)])
def test_no_stdout_on_refusal(arg, rc):
    p = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(HERE / "review_physical_snapshot.py"),
            str(EX),
            PIN if arg else "0" * 64,
            *arg,
        ],
        capture_output=True,
    )
    assert p.returncode == rc and p.stdout == b"" and p.stderr


def test_chart_regeneration():
    txt = chart.draw(EX, PIN, "walnut720")
    assert txt.encode() == (EX / "profile.svg").read_bytes()
    xml = ET.fromstring(txt)
    assert len(xml.find("{*}g/{*}polyline").attrib["points"].split()) == 512


def test_unsubmitted_plot_no_file(tmp_path):
    dst = tmp_path / "plot.svg"
    p = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(HERE / "plot_physical_profile.py"),
            str(EX),
            PIN,
            "pine90",
            str(dst),
        ],
        capture_output=True,
    )
    assert p.returncode == 1 and not dst.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "plan",
        "native",
        "astra",
        "order",
        "truncated",
        "extra",
        "slice-swap",
        "slice-bit",
        "missing",
        "duplicate-case",
        "extra-case",
        "unneeded-parent",
        "extra-catalog",
        "wrong-pin",
        "fake-complete",
        "orphan-as-report",
        "wrong-attempt",
        "rename-case",
        "wrong-grid",
        "wrong-vol",
        "harness",
        "protocol",
        "wrong-row-order",
        "negative-moment",
        "counter-type",
        "first-index",
        "wrong-baseline",
        "peak-plane-counter",
        "rebound-slice",
        "nan",
        "duplicate-json",
    ],
)
def test_resigned_corruption_cannot_bypass_independent_gates(fault):
    c = catalog()
    s = parts()
    rec = json.loads(s[3])
    q = json.loads(s[4])
    z = 256
    row = q["planes"][z]
    counts = row["gpu"]
    if fault in ("plan", "native", "astra"):
        s[("plan", "native", "astra").index(fault)] += b" "
    elif fault == "order":
        s[1], s[2] = s[2], s[1]
    elif fault == "truncated":
        s[-1] = s[-1][:-1]
    elif fault == "extra":
        s.append(b"x")
    elif fault == "slice-swap":
        s[5], s[6] = s[6], s[5]
    elif fault == "slice-bit":
        s[5] = bytes([s[5][0] ^ 1]) + s[5][1:]
    elif fault == "missing":
        s.pop(3)
    elif fault == "duplicate-case":
        c["case_commits"]["pine90"] = c["case_commits"]["walnut720"]
    elif fault == "extra-case":
        c["case_commits"]["walnut90"] = "1" * 64
    elif fault == "unneeded-parent":
        c["pine_parent_sha256"] = "0" * 64
    elif fault == "extra-catalog":
        c["total"] = 3
    elif fault == "wrong-pin":
        c["case_commits"]["walnut720"] = "0" * 64
    elif fault == "fake-complete":
        c["case_commits"].update(pine90="1" * 64, pine720="2" * 64)
    elif fault in ("orphan-as-report", "wrong-attempt", "rename-case"):
        if fault == "orphan-as-report":
            rec["report_sha256"] = (
                "28e4991cb067c1d43a5d691e169a807bfa272a3dd6f5b16b13e0d458ef296d0d"
            )
        elif fault == "wrong-attempt":
            rec["attempt"] = "pine90-" + "a" * 32
        else:
            rec["case"] = "pine90"
        s[3] = dump(rec)
        c["case_commits"]["walnut720"] = h(s[3])
    else:
        if fault == "wrong-grid":
            q["shape_xyz"][1] = 511
        if fault == "wrong-vol":
            q["source"]["pins"].reverse()
        if fault == "harness":
            q["harness_sha256"] = "a" * 64
        if fault == "protocol":
            q["protocol_sha256"] = "b" * 64
        if fault == "wrong-row-order":
            q["planes"][0], q["planes"][1] = q["planes"][1], q["planes"][0]
        if fault == "negative-moment":
            counts[10] = -1
        if fault == "counter-type":
            counts[5] = False
        if fault == "first-index":
            counts[7] += 1
            q["aggregate"]["first_maximum_xyz"][0] += 1
        if fault == "wrong-baseline":
            q["source"]["old"]["max_abs"] *= 2
        if fault == "peak-plane-counter":
            counts[2] -= 1
            q["aggregate"]["strict_exceedance"][0] -= 1
        if fault == "rebound-slice":
            s[5] = s[6]
            row["candidate_sha256"] = h(s[5])
        s[4] = dump(q)
        if fault == "nan":
            s[4] = s[4].replace(b'"scope":', b'"bad":NaN,"scope":')
        if fault == "duplicate-json":
            s[4] = s[4].replace(b'"scope":', b'"scope":null,"scope":')
        rec["report_sha256"] = h(s[4])
        s[3] = dump(rec)
        c["case_commits"]["walnut720"] = h(s[3])
    with pytest.raises(ValueError):
        bind(c, s)


def test_bounds_and_trailing_byte():
    s = frame(parts())
    c = catalog()
    for b in (s[:-1], s + b"x", rev.MAGIC + b"\xff" * 4, rev.MAGIC + b"\x00" * 4, b"bad"):
        c["pack_sha256"] = h(b)
        with pytest.raises(ValueError):
            rev.decode(c, b)


def test_hand_calculated_tie_two_planes():
    a = struct.pack("<4f", 1, 3, 0, 2)
    b = struct.pack("<4f", 1, 1, 2, 2)
    r = rev.rw.scalar(a, b)
    assert r == [4, 2, 2, 2, 2, 2, 2, 1, 0, 4, 8, 10]
    base = dict(candidate_sha256=h(a), reference_sha256=h(b), gpu=r)
    tot = rev.aggregate([dict(base, z=i) for i in range(2)], [2, 2, 2])
    assert (
        tot["first_maximum_xyz"] == [1, 0, 0]
        and tot["rel_l2"] == (16 / 20) ** 0.5
        and tot["numeric_equal"] == 4
    )


def test_empty_queue_and_external_pins(tmp_path):
    tmp_path.chmod(0o700)
    state = tmp_path / "queue"
    pack.queue.create(state)
    desc = pack.export(state, tmp_path / "out", [])
    d = rev.review(tmp_path / "out", desc["catalog_sha256"])
    assert d["committed_cases"] == d["specimens_with_evidence"] == 0 and all(
        "evidence" not in r for r in d["cases"]
    )
    for pins in ([("walnut720", "a" * 64)], [("walnut720", "a" * 64)] * 2):
        with pytest.raises(ValueError):
            pack.export(state, tmp_path / "never", pins)
        assert not (tmp_path / "never").exists()


def test_constructed_two_densities_one_object_not_physical_data():
    s = parts()
    c = catalog()
    n = 256
    a = struct.pack("<f", 2) * (n * n)
    b = struct.pack("<f", 1) * (n * n)
    stats = rev.rw.scalar(a, b)
    old = {"rel_l2": 1.0, "max_abs": 1.0}
    scan = dict(
        dense_sha256=h(a),
        repeat_sha256=h(a),
        astra_sha256=h(b),
        voxels=n**3,
        rois=[{}, {}],
        native_vs_astra=old,
    )
    p = dict(
        format="CFDK-Spark-physical-pine-qualification-v1",
        protocol_sha256=rev.PINE,
        source_manifest_sha256=rev.PINE_SOURCE,
        script_sha256=rev.PINE_TOOL,
        scans={
            case: dict(scan, views=views) for case, views in [("pine90", 90), ("pine720", 720)]
        },
    )
    par = dump(p)
    c["pine_parent_sha256"] = h(par)
    s.insert(3, par)
    for case in ("pine90", "pine720"):
        src = rev.expected_source(case, {"pine": p}, h(par))
        rows = [
            dict(
                z=z,
                gpu=stats,
                candidate_sha256=h(a) if not z else "1" * 64,
                reference_sha256=h(b) if not z else "2" * 64,
            )
            for z in range(n)
        ]
        q = dict(
            format="CFDK-paired-volume-qualification-v1",
            protocol_sha256=rev.rw.RULES,
            row_fields=rev.rw.FIELDS,
            build_sha256=rev.BUILD,
            harness_sha256=rev.HARNESS,
            witness_tool_sha256=rev.WITNESS,
            source=src,
            shape_xyz=[n] * 3,
            slices_verified_by_numpy=n,
            planes=rows,
            aggregate=rev.aggregate(rows, [n] * 3),
        )
        report = dump(q)
        rec = dump(
            dict(
                format="CFDK-verified-case-commit-v1",
                case=case,
                attempt=case + "-" + "a" * 32,
                report_sha256=h(report),
                pine_parent=h(par),
            )
        )
        c["case_commits"][case] = h(rec)
        s.extend([rec, report, a, b])
    d = bind(c, s)
    assert (
        d["committed_cases"] == 3 and d["specimens_with_evidence"] == 2 and d["all_cases_present"]
    )
    assert [r["evidence"]["peak_plane_voxels_remeasured"] for r in d["cases"]] == [
        512**2,
        256**2,
        256**2,
    ]
