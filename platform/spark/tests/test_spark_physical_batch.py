"""Plan accounting on mocked labels is not a physical result. The final tests use the real walnut."""

import itertools
import json
import struct
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import physical_batch as pb  # noqa: E402

NAMES = ("walnut720", "pine90", "pine720")
PINS = tuple(x * 64 for x in "abc")
WAL = "12f2f81c0d88d75b05fd824876b94cc236d63b5613260341c2917998a5db3349"
PACK = pb.qr.ROOT / "reports/spark-20261125/walnut-residual"


@pytest.mark.parametrize("mask", itertools.product((False, True), repeat=3))
def test_plan_only_each_subset(monkeypatch, mask):
    # Only the combinatorial state transitions are tested by this injected assessor.
    monkeypatch.setattr(pb, "assess", lambda case, path, sha, parent: {"mock_case": case})
    jobs = [(name, ".", pin) for name, pin, yes in zip(NAMES, PINS, mask, strict=True) if yes]
    par = "f" * 64 if any(mask[1:]) else None
    r = pb.census(jobs[::-1], par)
    assert (
        r["verified_cases"] == sum(mask)
        and r["specimens_with_evidence"] == int(mask[0]) + int(any(mask[1:]))
        and r["specimens_with_all_planned_cases"] == int(mask[0]) + int(all(mask[1:]))
    )
    assert [x["case"] for x in r["cases"]] == list(NAMES) and all(
        ("evidence" in x) == yes for x, yes in zip(r["cases"], mask, strict=True)
    )
    if all(mask):
        assert pb.census(jobs, par, True)["format"].endswith("complete-v1")
    else:
        with pytest.raises(pb.Incomplete):
            pb.census(jobs, par, True)


@pytest.mark.parametrize(
    "jobs,parent",
    [
        ([("other", ".", "a" * 64)], None),
        ([("walnut720", ".", "a" * 64)] * 2, None),
        ([("walnut720", ".", "a" * 64), ("pine90", ".", "a" * 64)], "c" * 64),
        ([("pine720", ".", "a" * 64)], None),
        ([], "c" * 64),
        ([("walnut720", ".", "not-a-hash")], None),
    ],
)
def test_bad_submission_never_assessed(monkeypatch, jobs, parent):
    def fail(*_):
        raise AssertionError("bad submission reached a source")

    monkeypatch.setattr(pb, "assess", fail)
    with pytest.raises(ValueError):
        pb.census(jobs, parent)


def analytic():
    def row(z, values):
        x = struct.pack("<2f", *values)
        y = struct.pack("<2f", 0.0, 0.0)
        return dict(
            z=z,
            candidate_sha256=pb.rw.hashed(x),
            reference_sha256=pb.rw.hashed(y),
            gpu=pb.rw.scalar(x, y),
        )

    return [row(0, (0.0, 1.0)), row(1, (-1.0, 0.0))]


def test_known_first_tie_zero_reference():
    a = pb.totals(analytic(), [2, 1, 2])
    assert (
        a["first_maximum_xyz"] == [1, 0, 0]
        and a["rel_l2"] is None
        and a["strict_exceedance"] == [2] * 4
        and a["numeric_equal"] == 2
        and a["signed_mean"] == 0
    )


@pytest.mark.parametrize(
    "kind",
    [
        "plane",
        "coverage",
        "bool",
        "impossible",
        "coords",
        "peak",
        "negative",
        "nonfinite",
        "digest",
    ],
)
def test_bad_record(kind):
    rows = analytic()
    r = rows[0]["gpu"]
    if kind == "plane":
        rows.reverse()
    if kind == "coverage":
        rows.pop()
    if kind == "bool":
        r[1] = True
    if kind == "impossible":
        r[3] = 2
    if kind == "coords":
        r[7] = 2
    if kind == "peak":
        r[6] = 0.0
    if kind == "negative":
        r[10] = -1.0
    if kind == "nonfinite":
        r[8] = float("inf")
    if kind == "digest":
        rows[1]["reference_sha256"] = "bad"
    with pytest.raises(ValueError):
        pb.totals(rows, [2, 1, 2])


def test_real_walnut_without_imputing_pine():
    r = pb.census([("walnut720", PACK, WAL)], None)
    assert [c["status"] for c in r["cases"]] == ["verified", "not_submitted", "not_submitted"]
    e = r["cases"][0]["evidence"]
    assert e["aggregate"]["voxels"] == 134217728 and e["remeasured_witness"]["maximum_xyz"] == [
        367,
        318,
        256,
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "repinned-aggregate",
        "repinned-parent",
        "repinned-case",
        "repinned-budget",
        "damaged-witness",
        "wrong-external-pin",
    ],
)
def test_real_walnut_forged_input_refused(tmp_path, kind):
    q = json.loads((PACK / "qualification.json").read_bytes())
    want = WAL
    if kind.startswith("repinned"):
        if kind == "repinned-aggregate":
            q["aggregate"]["numeric_equal"] += 1
        if kind == "repinned-parent":
            q["source"]["parents"]["native"] = "c" * 64
        if kind == "repinned-case":
            q["source"]["case"] = "pine720"
        if kind == "repinned-budget":
            q["accounting"][1]["owned_bytes"] += 1
        buf = json.dumps(q).encode()
        want = pb.rw.hashed(buf)
        (tmp_path / "qualification.json").write_bytes(buf)
    else:
        (tmp_path / "qualification.json").hardlink_to(PACK / "qualification.json")
    if kind == "wrong-external-pin":
        want = "a" * 64
    for n in ("candidate-slice.f32", "reference-slice.f32"):
        if kind == "damaged-witness" and n.startswith("candidate"):
            (tmp_path / n).write_bytes(b"junk")
        else:
            (tmp_path / n).hardlink_to(PACK / n)
    with pytest.raises(ValueError):
        pb.census([("walnut720", tmp_path, want)], None)
