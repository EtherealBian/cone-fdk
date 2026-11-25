"""Durable case-level execution on trusted, private local Spark storage. A published result without a commit is never adopted."""

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

import physical_batch as pb
from sealed_inputs import publish_new, syncdir
from spark_publication import anchored

HERE = Path(__file__).resolve().parent
M = HERE.parents[2]
CASES = ("walnut720", "pine90", "pine720")
CODE = {
    "physical_batch.py": "0f4251d39192d3fbb2a55c1005a0956d40360ed9ea2dde5c149491871c14d288",
    "qualify_residual.py": "2f21f3c7a2d517b5580567cd2feae38d5090a7958f420d67a92a2f0c735f350c",
    "residual_witness.py": "b127c8be1a94e96a24e9a1404e5923a7bfdd8b2a7cf034db4e6ee76f4d6bafef",
    "build_residual.py": "b1cc2561fddda07c4bc9f56f31a4739fef794a5443871cda12916ad0a1987ab2",
    "sealed_inputs.py": "16fc3e8b791d6aab57a204781e124b4d2eb8a1b98082db1e0392fddadffbdec2",
    "spark_publication.py": "a497b5387b56c602191da289b2dca36fbff9d58465a9c1e7060abca7a76dfa66",
}
need = pb.need


def ident():
    for n, k in CODE.items():
        need(pb.qr.sha(HERE / n) == k, "changed queue dependency: " + n)
    plan = pb.qr.json_checked(pb.qr.ROOT / "fixtures/physical-residual-batch-v1.json", pb.PLAN)
    need(
        tuple(c["case"] for c in plan["cases"]) == CASES,
        "queue case selection differs from frozen plan",
    )
    return dict(
        format="CFDK-physical-run-queue-v1",
        driver_sha256=pb.qr.sha(__file__),
        plan_sha256=pb.PLAN,
        worker_build=pb.qr.BUILD,
        code=CODE,
    )


def encoded(v):
    return (json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def private(p, dir=True):
    s = p.lstat()
    need(
        (stat.S_ISDIR if dir else stat.S_ISREG)(s.st_mode)
        and s.st_uid == os.getuid()
        and not s.st_mode & 0o077
        and (dir or s.st_nlink == 1)
        and p == p.resolve(strict=True),
        "private real state required",
    )


def owned_json(p):
    private(p, False)
    need(0 < p.stat().st_size <= 16384, "oversize queue metadata")
    data = p.read_bytes()
    v = json.loads(data)
    need(encoded(v) == data, "noncanonical state record")
    return v


def write(p, v):
    with p.open("xb") as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(encoded(v))
        f.flush()
        os.fsync(f.fileno())


def commit_new(old, new):
    """Move one fully fsynced record. No replace, no dual-hardlink crash window."""
    func = ctypes.CDLL(None, use_errno=True).renameat2
    func.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    func.restype = ctypes.c_int
    if func(-100, os.fsencode(old), -100, os.fsencode(new), 1):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(new))
    syncdir(new.parent)
    syncdir(old.parent)


def create(state):
    state = Path(state).absolute()
    private(state.parent)
    identity = ident()
    with (
        anchored(state) as target,
        tempfile.TemporaryDirectory(dir=target.parent, prefix=".casequeue-init-") as t,
    ):
        stage = Path(t) / "new"
        stage.mkdir(mode=0o700)
        for name in ("commits", "attempts"):
            (stage / name).mkdir(mode=0o700)
            syncdir(stage / name)
        write(stage / "identity.json", identity)
        publish_new(stage, target)
    return identity


@contextmanager
def locked(state):
    private(state.parent)
    private(state)
    fd = os.open(state, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(os.path.samestat(os.fstat(fd), state.stat()), "queue moved")
        yield fd
    finally:
        os.close(fd)


def attempt_name(name):
    return re.fullmatch(r"(walnut720|pine90|pine720)-[0-9a-f]{32}", name) is not None


def authenticated(case, folder, k, parent):
    need(folder == folder.resolve(strict=True) and folder.is_dir(), "aliased worker directory")
    names = {"qualification.json", "candidate-slice.f32", "reference-slice.f32", "witness.json"}
    need(
        {p.name for p in folder.iterdir()} == names
        and all(stat.S_ISREG((folder / n).lstat().st_mode) for n in names),
        "unexpected worker outputs",
    )
    evidence = pb.assess(case, folder, k, parent)
    w = folder / "witness.json"
    need(
        0 < w.stat().st_size <= 65536
        and pb.same(json.loads(w.read_bytes()), evidence["remeasured_witness"]),
        "worker witness record differs from remeasurement",
    )
    return evidence


def worker(case, run, parent, guard):
    # The child retains the queue lock if its dispatcher is killed, and flock retains the GPU lock until the qualifier exits.
    cmd = [
        "flock",
        "-w",
        "300",
        str(M / "gpu.lock"),
        sys.executable,
        "-B",
        str(HERE / "qualify_residual.py"),
        case,
        str(M / "residual-builds-001/qualified"),
        str(run / "result"),
    ]
    if parent is not None:
        cmd += ["--pine-pin", parent]
    with (run / "worker.stdout").open("xb") as out, (run / "worker.stderr").open("xb") as err:
        proc = subprocess.run(
            cmd,
            stdout=out,
            stderr=err,
            pass_fds=(guard,),
            cwd=HERE,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        out.flush()
        err.flush()
        os.fsync(out.fileno())
        os.fsync(err.fileno())
    need(proc.returncode == 0, "worker failed; attempt and diagnostics retained: " + run.name)
    p = run / "worker.stdout"
    need(0 < p.stat().st_size <= 16384, "worker ack size")
    ack = json.loads(p.read_bytes())
    return pb.rw.pin(ack["report_sha256"])


def execute(state, parent=None, *, work=False, limit=3, after=None):
    state = Path(state).absolute()
    need(type(limit) is int and 1 <= limit <= 3, "case limit")
    identity = ident()
    commits = state / "commits"
    attempts = state / "attempts"
    done = {}
    items = []
    used = set()
    fresh = 0
    with locked(state) as fd:
        need(
            {x.name for x in state.iterdir()} == {"identity.json", "attempts", "commits"}
            and owned_json(state / "identity.json") == identity,
            "different queue identity",
        )
        private(commits)
        private(attempts)
        names = {x.name for x in commits.iterdir()}
        need(names <= {n + ".json" for n in CASES}, "unknown commit entry")
        for a in attempts.iterdir():
            private(a)
            need(attempt_name(a.name), "invalid attempt directory")
        if parent is not None:
            pb.rw.pin(parent)
            for case in CASES[1:]:
                pb.qr.volumes(case, M, parent)

        def inspect(case, record):
            name = record["attempt"]
            k = pb.rw.pin(record["report_sha256"])
            par = parent if case.startswith("pine") else None
            need(
                set(record) == {"case", "attempt", "report_sha256", "pine_parent", "format"}
                and record["format"] == "CFDK-verified-case-commit-v1"
                and record["case"] == case
                and record["pine_parent"] == par
                and (par is not None or case == CASES[0])
                and attempt_name(name)
                and name.startswith(case + "-")
                and name not in used,
                "invalid case commit or missing external parent",
            )
            run = attempts / name
            private(run)
            e = authenticated(case, run / "result", k, par)
            used.add(name)
            done[case] = (record, e)

        # Recheck every committed case before starting even one fresh job.
        for case in CASES:
            if case + ".json" in names:
                inspect(case, owned_json(commits / (case + ".json")))
        for case in CASES:
            if case in done:
                kind = "reused"
            elif case.startswith("pine") and parent is None:
                kind = "awaiting_external_parent"
            elif not work or fresh >= limit:
                kind = "ready_not_run"
            else:
                par = parent if case.startswith("pine") else None
                run = attempts / (case + "-" + uuid.uuid4().hex)
                run.mkdir(mode=0o700)
                syncdir(attempts)
                k = worker(case, run, par, fd)
                record = dict(
                    format="CFDK-verified-case-commit-v1",
                    case=case,
                    attempt=run.name,
                    report_sha256=k,
                    pine_parent=par,
                )
                inspect(case, record)
                pending = run / "pending.json"
                write(pending, record)
                syncdir(run)
                if after is not None:
                    after("verified_before_commit", case)
                commit_new(pending, commits / (case + ".json"))
                if after is not None:
                    after("committed", case)
                kind = "fresh"
                fresh += 1
            row = dict(case=case, execution=kind)
            if case in done:
                r, e = done[case]
                row.update(attempt=r["attempt"], report_sha256=r["report_sha256"], evidence=e)
            items.append(row)
        return dict(
            format="CFDK-physical-queue-progress-v1",
            identity_sha256=hashlib.sha256(encoded(identity)).hexdigest(),
            cases=items,
            committed_cases=len(done),
            fresh_cases=fresh,
            orphan_attempts=sorted(p.name for p in attempts.iterdir() if p.name not in used),
            scope="Execution checkpoint, not external final acceptance. No metrics or case commits for missing prerequisites. No automatic promotion or deletion of abandoned attempts. Timing-bearing new worker receipts need their own external pins.",
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["init", "status", "work"])
    p.add_argument("state", type=Path)
    p.add_argument("--pine-parent")
    p.add_argument("--limit", type=int, default=3)
    a = p.parse_args()
    try:
        if a.mode == "init":
            need(not a.pine_parent and a.limit == 3, "irrelevant init options")
            r = create(a.state)
        else:
            r = execute(a.state, a.pine_parent, work=a.mode == "work", limit=a.limit)
        print(encoded(r).decode(), end="")
    except (ValueError, KeyError, TypeError, IndexError, OSError, RuntimeError) as e:
        p.exit(1, f"physical queue refused: {e}\n")


if __name__ == "__main__":
    main()
