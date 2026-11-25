"""Durable single-writer batch of detached review jobs on trusted, local Linux storage.

Fresh state is private. All sessions reauthenticate every logical payload incl. reused jobs.
SQLite is a local checkpoint, not an attestation against a malicious same-account writer.
No final receipt until all rows, withheld inventory, total cells and EOF pass.
"""

import fcntl
import os
import sqlite3
import stat
import tempfile
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import offline_campaign as oc
from sealed_inputs import syncdir
from spark_publication import anchored


def file_sha(path):
    return oc.sha(Path(path).read_bytes())


def origin(chain):
    return oc.encoded(
        dict(
            format="CFDK-local-periodic-checkpoints-v1",
            chain_sha256=oc.pin(chain),
            reviewer_sha256=file_sha(oc.__file__),
            driver_sha256=file_sha(__file__),
        )
    )


def private(path, is_dir):
    info = path.lstat()
    oc.need(
        (stat.S_ISDIR if is_dir else stat.S_ISREG)(info.st_mode)
        and info.st_uid == os.getuid()
        and not info.st_mode & 0o077
        and (is_dir or info.st_nlink == 1),
        "private state required",
    )


@contextmanager
def opened(state, resume):
    state = Path(state).absolute()
    oc.need(
        state == state.resolve(strict=False) and state.parent.is_dir(),
        "no alias/traversal in state path",
    )
    if not resume:
        state.mkdir(mode=0o700)
        syncdir(state.parent)
    private(state, True)
    # Lock the directory inode: no empty lockfile on interrupted startup.
    guard = os.open(state, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = Path(f"/proc/{os.getpid()}/fd/{guard}")
        dbfile = root / "jobs.sqlite3"
        allowed = {"jobs.sqlite3", "jobs.sqlite3-journal"}
        for member in root.iterdir():
            oc.need(member.name in allowed, "unexpected entry in checkpoint")
            private(member, False)
        if not resume:
            fd = os.open(dbfile, os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_RDWR, 0o600)
            os.close(fd)
        else:
            private(dbfile, False)
        db = sqlite3.connect(f"file:{dbfile}?mode=rw", uri=True, timeout=0)
        try:
            oc.need(
                db.execute("pragma journal_mode=DELETE").fetchone() == ("delete",),
                "local journal mode",
            )
            db.execute("pragma synchronous=FULL")
            db.execute("pragma trusted_schema=OFF")
            if not resume:
                with db:
                    db.execute(
                        "create table identity (one integer primary key check(one=1), value blob not null)"
                    )
                    db.execute(
                        "create table jobs (job text primary key, input text not null, receipt blob not null)"
                    )
                syncdir(root)
            oc.need(
                db.execute("pragma quick_check").fetchall() == [("ok",)],
                "checkpoint database damaged",
            )
            yield db
        finally:
            db.close()
    finally:
        os.close(guard)


def publish(path, data):
    with anchored(path) as target:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".review-") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            os.link(f.name, target)
            syncdir(target.parent)


def batch(pack, chain, state, out, *, resume=False, after_commit=None):
    pack, state, out = (Path(x).absolute() for x in (pack, state, out))
    oc.pin(chain)
    oc.need(
        not out.is_relative_to(state) and not pack.is_relative_to(state) and out != pack,
        "checkpoint, source and published receipt must be separate",
    )
    # Reject an existing output before creating state or doing work. Final publish repeats this.
    with anchored(out):
        pass
    expected = origin(chain)
    reused = fresh = 0
    seen = set()
    with opened(state, resume) as db:
        if not resume:
            with db:
                db.execute("insert into identity values(1,?)", (expected,))
        oc.need(
            db.execute("select one,value from identity").fetchall() == [(1, expected)],
            "different chain or verifier version",
        )

        def checkpoint(row, result, compute):
            nonlocal reused, fresh
            case, a, b = oc.ident(row)
            key = f"{case}:{a}:{b}"
            seen.add(key)
            digest = oc.sha(oc.encoded(row))
            data = oc.encoded(result)
            saved = db.execute("select input,receipt from jobs where job=?", (key,)).fetchone()
            if saved is not None:
                oc.need(saved == (digest, data), "damaged or mismatched committed checkpoint")
                reused += 1
                return
            compute()
            with db:
                db.execute("insert into jobs values(?,?,?)", (key, digest, data))
            fresh += 1
            if after_commit is not None:
                after_commit(fresh)

        result = oc.review(pack, chain, checkpoint=checkpoint)
        oc.need(
            {r[0] for r in db.execute("select job from jobs")} == seen,
            "unexpected cached job, possibly withheld",
        )
        data = oc.encoded(result) + b"\n"
        publish(out, data)
    return dict(
        format="CFDK-batch-review-run-v1",
        chain_sha256=chain,
        receipt_sha256=oc.sha(data),
        fresh=fresh,
        reused=reused,
        refusals=len(result["refused"]),
        cells=result["cells"],
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("pack", type=Path)
    p.add_argument("chain_sha256")
    p.add_argument("state", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    try:
        print(
            oc.encoded(
                batch(args.pack, args.chain_sha256, args.state, args.output, resume=args.resume)
            ).decode("ascii")
        )
    except (
        ValueError,
        OSError,
        sqlite3.Error,
        KeyError,
        IndexError,
        TypeError,
        RecursionError,
    ) as e:
        p.exit(1, f"batch refused: {e}\n")
