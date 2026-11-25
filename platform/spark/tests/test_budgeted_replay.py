"""Explicit device memory limits must hold for order, counters, sparse resumes and real cached bytes."""

import functools
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import budgeted_replay as b
import review_budget_frontier as v
import qualify_budgeted_replay as qb
from grouped_replay import sr, old

PIN = "0bae9a10cfd9770acd31f2b871c588509d8a68276c0f7af7868b6c4b3dcc6234"


def plan():
    return v.load(b.g.FIXTURE, b.g.PLAN)


def qual():
    return v.load(b.M / "gb10-budget-frontier-002/qualification.json", PIN)


def parent():
    return v.load(b.M / "grouped-replay-qualified-001/qualification.json", b.CERT)


@pytest.mark.parametrize(
    "cap,num",
    [
        (1, None),
        (160, None),
        (161, 5),
        (216, 5),
        (217, 4),
        (252, 4),
        (253, 3),
        (325, 3),
        (326, 2),
        (554, 2),
        (555, 1),
        (768, 1),
    ],
)
def test_boundaries(cap, num):
    if num is None:
        with pytest.raises(ValueError, match="largest missing"):
            b.make_plan(plan(), [], cap)
    else:
        q = b.make_plan(plan(), [], cap)
        assert q["jobs"] == num and q["peak_owned_bytes"] + 512 <= cap * (1 << 20)


@pytest.mark.parametrize("saved", [[0, 0], [True], {999}, [73.0], None, "", iter([0]), [[0]]])
def test_refuse_ambiguous_saved(saved):
    with pytest.raises(ValueError, match="cache keys"):
        b.make_plan(plan(), saved, 555)


@pytest.mark.parametrize("cap", [True, False, 0, 1.5, "555", 769])
def test_refuse_bad_cap(cap):
    with pytest.raises(ValueError, match="integer"):
        b.make_plan(plan(), [], cap)


def test_crosscheck_all_subsets_and_mib_limits():
    d = plan()
    q = qual()
    status = qb.oracle(d)
    assert (
        status
        == q["independent"]
        == v.Model(d, parent()["decision"]["grouped_peak_bytes"] - 4 * d["n"] ** 3).census
    )
    assert status["accepted"] + status["refused"] == 24576


def test_deterministic_noncontiguous():
    d = plan()
    m = b.make_plan(d, [], 253)
    assert [g["z"] for g in m["groups"]] == [[0, 283], [73, 174], [401]]
    for saved in ({0, 283}, [283, 0], {0: 1, 283: 1}):
        assert b.make_plan(d, saved, 161)["reused_z"] == [0, 283]
    full = b.make_plan(d, [r["z"] for r in d["slabs"]], 1)
    assert full["groups"] == [] and full["peak_owned_bytes"] == full["jobs"] == 0


@pytest.fixture(scope="module")
def census_once():
    return v.Model(plan(), 44114048).census


@pytest.mark.parametrize(
    "fault",
    [
        "run-order",
        "missing-run",
        "capacity",
        "schedule",
        "tie",
        "subgroup",
        "jobs",
        "offset",
        "format",
        "batch",
        "projected",
        "actual",
        "bool",
        "reserve",
        "host",
        "fft",
        "source",
        "missing-step",
        "extra-work",
        "partial",
        "completed",
        "zero",
        "duplicate",
        "refusal",
        "manifest",
        "volume",
        "timer",
        "census",
        "parent",
    ],
)
def test_detached_rejects_tampering(fault, monkeypatch, census_once):
    # Cache only the exhaustive enumeration in these fault-injection cases. The integration test below runs it unpatched.
    monkeypatch.setattr(v.Model, "census", census_once)
    q = qual()
    r = q["runs"][2]
    w = r["phases"][0]
    n = w["issued"][0]["native"]
    mixed = q["changed_caps"]
    g = w["schedule"]["groups"][0]
    p = parent()
    if fault == "run-order":
        q["runs"].reverse()
    elif fault == "missing-run":
        q["runs"].pop()
    elif fault == "capacity":
        n["effective_budget_bytes"] += 1
    elif fault == "schedule":
        w["schedule"]["feasible_partitions"] -= 1
    elif fault == "tie":
        g["z"].reverse()
    elif fault == "subgroup":
        w["issued"][0]["z"] = [0, 174]
    elif fault == "jobs":
        w["max_jobs"] = True
    elif fault == "offset":
        n["regions"][1]["offset_bytes"] += 4
    elif fault == "format":
        n["format"] = "dense"
    elif fault == "batch":
        n["requested_batch"] = 8
    elif fault == "projected":
        w["schedule"]["projected_projection_bytes"] += 1
    elif fault == "actual":
        w["actual_projection_bytes"] -= 1
    elif fault == "bool":
        n["feasible_plans_scored"][0]["passes"] = True
    elif fault == "reserve":
        g["planner_min_bytes"] -= 512
    elif fault == "host":
        n["host_pinned_bytes"] = False
    elif fault == "fft":
        n["fft_workspace_bytes"] = False
    elif fault == "source":
        w["source_sha256"] = "0" * 64
    elif fault == "missing-step":
        mixed["phases"].pop(1)
    elif fault == "extra-work":
        mixed["phases"][-1]["issued"] = w["issued"]
    elif fault == "partial":
        mixed["phases"][1]["present_z"].pop()
    elif fault == "completed":
        mixed["final_workers"] = False
    elif fault == "zero":
        mixed["phases"][-1]["actual_peak_owned_bytes"] = 1
    elif fault == "duplicate":
        mixed["phases"][2]["schedule"]["reused_z"].append(0)
    elif fault == "refusal":
        q["refused_cold_mib"] = 161
    elif fault == "manifest":
        r["manifest"] = "1" * 64
    elif fault == "volume":
        r["volume"] = "1" * 64
    elif fault == "timer":
        n["wall_ms"] = float("inf")
    elif fault == "census":
        q["independent"]["accepted"] -= 1
    elif fault == "parent":
        p["decision"]["gate"] = False
    with pytest.raises(ValueError):
        v.check(q, plan(), p)


def test_detached_certificate_and_fresh_production_source():
    q = qual()
    assert q["source_sha256"] == {
        p.name: b.g.sha(p) for p in (Path(b.__file__), Path(qb.__file__))
    }
    result = v.check(q, plan(), parent())
    assert result == json.loads(
        (b.M / "gb10-budget-frontier-002/detached-review.json").read_bytes()
    )
    assert (result["verified_native_workers"], result["verified_deliveries"]) == (18, 6)


@pytest.mark.parametrize(
    "fault",
    [
        "corrupt",
        "bad-fft",
        "bad-budget",
        "bad-peak",
        "bool-planner",
        "nonfresh",
        "overlap",
        "symlink",
        "traversal",
        "cold160",
    ],
)
def test_no_false_receipt_or_cuda_for_bad_input(tmp_path, monkeypatch, fault):
    wd = tmp_path / "owner"
    wd.mkdir(mode=0o700)
    cache = wd / "cache"
    out = wd / "out"
    d = plan()
    cap = 161
    launches = []
    if fault == "corrupt":
        old.validate_cache(cache)
        s = d["slabs"][0]
        size = sr.size(d, s)
        with closing(sqlite3.connect(cache / "slabs.sqlite3")) as db:
            sr.attach(db, b.g.PLAN, {s["z"]: size})
            with db:
                db.execute("insert into slabs values(?,zeroblob(?))", (s["z"], size))
    if fault == "nonfresh":
        out.touch()
    if fault == "overlap":
        out = cache
    if fault == "symlink":
        cache.symlink_to(wd / "absent")
    if fault == "traversal":
        cache = wd / ".." / "owner" / "cache"
    if fault == "cold160":
        cap = 160

    def fake(*args):
        launches.append(1)
        assert fault in {"bad-fft", "bad-budget", "bad-peak", "bool-planner"}
        n = qual()["runs"][0]["phases"][0]["issued"][0]["native"]
        if fault == "bad-fft":
            n["fft_workspace_bytes"] = False
        if fault == "bad-budget":
            n["effective_budget_bytes"] += 1
        if fault == "bad-peak":
            n["peak_owned_bytes"] -= 1
        if fault == "bool-planner":
            n["feasible_plans_scored"][0]["passes"] = True
        return n, 10

    monkeypatch.setattr(b.g, "native", fake)
    with pytest.raises((OSError, ValueError)):
        b.warm(cache, out, cap)
    assert bool(launches) == (fault in {"bad-fft", "bad-budget", "bad-peak", "bool-planner"})
    if out.exists():
        assert fault == "nonfresh" and out.is_file() and out.stat().st_size == 0
    if fault == "cold160":
        assert not list(wd.iterdir())
    if launches:
        with closing(sqlite3.connect(cache / "slabs.sqlite3")) as db:
            assert db.execute("select count(*) from slabs").fetchone() == (0,)
