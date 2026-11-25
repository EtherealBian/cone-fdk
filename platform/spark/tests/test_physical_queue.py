"""Protocol/crash edge cases; fake workers below are not measured acquisitions."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import physical_queue as q  # noqa: E402


@pytest.fixture
def state(tmp_path):
    tmp_path.chmod(0o700)
    s = tmp_path / "queue"
    q.create(s)
    return s


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def worker(case, run, parent, guard):
        assert os.fstat(guard)
        seen.append(case)
        (run / "result").mkdir()
        raw = case.encode()
        (run / "result" / "fake").write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def check(case, folder, pin, parent):
        assert (
            folder.resolve(strict=True) == folder
            and (folder / "fake").read_bytes() == case.encode()
            and hashlib.sha256(case.encode()).hexdigest() == pin
        )
        return {"simulated_protocol_test": True}

    monkeypatch.setattr(q, "worker", worker)
    monkeypatch.setattr(q, "authenticated", check)
    monkeypatch.setattr(q.pb.qr, "volumes", lambda *_: {})
    return seen


def test_stepwise_unblock_by_external_parent(state, calls):
    p = "b" * 64
    a = q.execute(state, work=True)
    assert calls == ["walnut720"] and [r["execution"] for r in a["cases"]] == [
        "fresh",
        "awaiting_external_parent",
        "awaiting_external_parent",
    ]
    assert all("evidence" not in r for r in a["cases"][1:])
    b = q.execute(state, p, work=True, limit=1)
    assert [r["execution"] for r in b["cases"]] == ["reused", "fresh", "ready_not_run"]
    assert q.execute(state, p, work=True)["committed_cases"] == 3 and calls == list(q.CASES)
    assert q.execute(state, p, work=True)["fresh_cases"] == 0 and len(calls) == 3


@pytest.mark.parametrize("p", [None, "a" * 64, "not-a-pin"])
def test_pine_commit_requires_original_external_parent(state, calls, p):
    q.execute(state, "b" * 64, work=True)
    calls.clear()
    with pytest.raises(ValueError):
        q.execute(state, p, work=True)
    assert not calls


@pytest.mark.parametrize("phase", ["verified_before_commit", "committed"])
def test_interrupt_and_retry(state, calls, phase):
    class Crashed(Exception):
        pass

    def crash(at, case):
        if at == phase:
            raise Crashed

    with pytest.raises(Crashed):
        q.execute(state, work=True, after=crash)
    result = q.execute(state, work=True)
    assert calls == ["walnut720"] * (2 if phase == "verified_before_commit" else 1)
    assert result["committed_cases"] == 1 and len(result["orphan_attempts"]) == (
        phase == "verified_before_commit"
    )
    assert not list((state / "commits").glob("*.tmp"))


@pytest.mark.parametrize(
    "hazard",
    [
        "identity",
        "extra-dir",
        "external-dir",
        "commit-link",
        "unknown-case",
        "case-mismatch",
        "traversal",
        "wrong-prefix",
        "oversized",
        "not-private",
        "duplicate-json",
        "orphan-name",
        "attempt-alias",
        "extra-commit-field",
        "commit-hardlink",
    ],
)
def test_tamper_rejected_before_worker(state, calls, hazard):
    doc = q.execute(state, work=True)
    calls.clear()
    row = doc["cases"][0]
    rec = state / "commits/walnut720.json"
    v = json.loads(rec.read_bytes())
    alias = state / "attempts" / row["attempt"]
    other = state.parent / "outside"
    if hazard == "identity":
        (state / "identity.json").write_text("{}")
    if hazard == "extra-dir":
        (state / "extra").mkdir()
    if hazard == "external-dir":
        (state / "commits").rename(other)
        (state / "commits").symlink_to(other, target_is_directory=True)
    if hazard == "unknown-case":
        (state / "commits/bogus.json").touch()
    if hazard == "not-private":
        rec.chmod(0o644)
    if hazard == "oversized":
        rec.write_bytes(b"x" * 16385)
    if hazard == "duplicate-json":
        rec.write_bytes(rec.read_bytes().replace(b"{", b'{"case":"walnut720",', 1))
    if hazard == "commit-link":
        rec.rename(other)
        rec.symlink_to(other)
    if hazard == "commit-hardlink":
        os.link(rec, other)
    if hazard == "orphan-name":
        (state / "attempts/unknown").mkdir(mode=0o700)
    if hazard == "attempt-alias":
        alias.rename(other)
        alias.symlink_to(other, target_is_directory=True)
    if hazard in ["case-mismatch", "traversal", "wrong-prefix", "extra-commit-field"]:
        if hazard == "case-mismatch":
            v["case"] = "pine90"
        if hazard == "traversal":
            v["attempt"] = "../elsewhere"
        if hazard == "wrong-prefix":
            v["attempt"] = "pine720-" + "a" * 32
        if hazard == "extra-commit-field":
            v["unused"] = True
        rec.write_bytes(q.encoded(v))
    with pytest.raises((OSError, ValueError)):
        q.execute(state, "b" * 64, work=True)
    assert not calls


def test_nonreplacing_private_init_and_guard(state, calls):
    with pytest.raises(ValueError):
        q.create(state)
    with q.locked(state), pytest.raises(BlockingIOError):
        q.execute(state, work=True)
    assert not calls


def test_noreplace_keeps_both_files(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_text("pending")
    b.write_text("already committed")
    with pytest.raises(FileExistsError):
        q.commit_new(a, b)
    assert a.read_text() == "pending" and b.read_text() == "already committed"


@pytest.mark.parametrize("limit", [0, 4, True, 1.0])
def test_bad_limit(state, calls, limit):
    with pytest.raises(ValueError):
        q.execute(state, work=True, limit=limit)
    assert not calls


def test_failed_worker_does_not_commit_and_leaves_diagnostic(state, monkeypatch):
    def fail(_case, run, _parent, _guard):
        (run / "worker.stderr").write_text("fail marker")
        raise ValueError("reject")

    monkeypatch.setattr(q, "worker", fail)
    with pytest.raises(ValueError, match="reject"):
        q.execute(state, work=True)
    s = q.execute(state)
    assert s["committed_cases"] == 0 and len(s["orphan_attempts"]) == 1
    assert (
        state / "attempts" / s["orphan_attempts"][0] / "worker.stderr"
    ).read_text() == "fail marker"


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "result-alias",
        "aliased-slice",
        "witness-change",
        "missing-witness",
        "wrong-qualification-pin",
        "control",
    ],
)
def test_real_walnut_receipt_gate(tmp_path, damage):
    src = q.M / "residual-walnut-001"
    p = tmp_path / "result"
    p.mkdir()
    k = "12f2f81c0d88d75b05fd824876b94cc236d63b5613260341c2917998a5db3349"
    for name in (
        "reference-slice.f32",
        "candidate-slice.f32",
        "qualification.json",
        "witness.json",
    ):
        os.link(src / name, p / name)
    if damage == "extra":
        (p / "extra").touch()
    if damage == "result-alias":
        p.rename(tmp_path / "elsewhere")
        p.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    if damage == "aliased-slice":
        (p / "reference-slice.f32").unlink()
        (p / "reference-slice.f32").symlink_to(src / "reference-slice.f32")
    if damage in ("witness-change", "missing-witness"):
        (p / "witness.json").unlink()
    if damage == "witness-change":
        (p / "witness.json").write_text("{}")
    if damage == "wrong-qualification-pin":
        k = "c" * 64
    if damage == "control":
        assert q.authenticated("walnut720", p, k, None)["aggregate"]["voxels"] == 134217728
    else:
        with pytest.raises((ValueError, OSError)):
            q.authenticated("walnut720", p, k, None)
