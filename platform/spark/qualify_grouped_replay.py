"""Time cold serial vs grouped durable replay, then kill a real worker at two transaction boundaries."""

import argparse
import hashlib
import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path
from statistics import median
import grouped_replay as g
from grouped_replay import old, sr, need, M
from spark_publication import fresh, publish, staging

HERE = Path(__file__).resolve().parent


def offline(dest, d):
    p = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(g.ROOT / "scripts/slab_replay.py"),
            "verify",
            str(g.FIXTURE),
            str(dest),
            "--plan-pin",
            g.PLAN,
        ],
        capture_output=True,
        timeout=60,
    )
    need(
        not p.returncode and not p.stderr and json.loads(p.stdout) == sr.manifest(d, g.PLAN),
        "detached stdlib validation",
    )
    return g.sha(dest / "manifest.json")


def trial(kind, job, d, cache=None, reuse=()):
    cache = cache or job / "cache"
    dst = job / "delivery"
    reports = []

    def runner(*args):
        r = old.native(*args)
        reports.append(r)
        return r

    start = time.monotonic_ns()
    if kind == "serial":
        r = old.generate(g.FIXTURE, g.PLAN, g.SCAN, g.BUILD, cache, dst, runner=runner)
        rows = reports
        need(r["reused"] == len(reuse), "serial resumed count")
    else:
        need(kind == "grouped", "unknown arm")
        r = g.issue(cache, dst)
        need(r["reused_z"] == list(reuse), "bad reuse")
        rows = [q["native"] for q in r["groups"]]
    dt = (time.monotonic_ns() - start) / 1e6
    mf = offline(dst, d)
    need(
        mf == r["manifest_sha256"] and r["volume_sha256"] == d["volume_sha256"],
        "different reconstruction",
    )
    missing = [s["z"] for s in d["slabs"] if s["z"] not in reuse]
    expected = len(missing) if kind == "serial" else int(bool(missing))
    need(len(rows) == expected, "wrong number of native launches")
    need(
        all(
            q["gpu"] == "NVIDIA GB10"
            and q["redzones_verified"] is True
            and q["filter_batches"] == 45
            and q["projection_bytes_read"] == d["projections_bytes"]
            and q["projection_passes"] == 1
            and 0 < q["peak_owned_bytes"] <= q["budget_bytes"] == 768 * g.MIB
            for q in rows
        ),
        "work counts",
    )
    need(sum(q["launches_bp"] for q in rows) == 45 * len(missing), "wrong backprojections")
    # Cross-check every closed checkpoint before allowing this private trial to discard it.
    with closing(sqlite3.connect(f"file:{cache}/slabs.sqlite3?mode=ro", uri=True)) as db:
        need(
            db.execute("select z,length(payload) from slabs order by z").fetchall()
            == [(s["z"], sr.size(d, s)) for s in d["slabs"]],
            "incomplete checkpoint",
        )
        for s in d["slabs"]:
            with db.blobopen("slabs", "payload", s["z"], readonly=True) as blob:
                sr.transfer(blob, sr.size(d, s), s["sha256"])
    return dict(
        arm=kind,
        end_to_end_ms=dt,
        reused_z=list(reuse),
        computed_z=missing,
        workers=len(rows),
        filter_batches=sum(q["filter_batches"] for q in rows),
        projection_bytes=sum(q["projection_bytes_read"] for q in rows),
        bp_launches=sum(q["launches_bp"] for q in rows),
        peak_owned_bytes=max((q["peak_owned_bytes"] for q in rows), default=0),
        volume_sha256=r["volume_sha256"],
        manifest_sha256=mf,
        report=r,
    )


def balanced():
    for j in range(6):
        for mode in ("serial", "grouped") if j % 2 == 0 else ("grouped", "serial"):
            yield j, mode


def decision(rows, d):
    need([(r["block"], r["arm"]) for r in rows] == list(balanced()), "incomplete paired schedule")
    hs = hashlib.sha256(sr.encode(sr.manifest(d, g.PLAN))).hexdigest()
    zs = [s["z"] for s in d["slabs"]]
    for r in rows:
        w = 5 if r["arm"] == "serial" else 1
        need(
            r["workers"] == w
            and r["filter_batches"] == 45 * w
            and r["projection_bytes"] == w * d["projections_bytes"]
            and r["bp_launches"] == 225
            and r["volume_sha256"] == d["volume_sha256"]
            and r["manifest_sha256"] == hs
            and r["computed_z"] == zs
            and r["reused_z"] == []
            and 0 < r["peak_owned_bytes"] <= 768 * g.MIB
            and type(r["end_to_end_ms"]) in (float, int)
            and math.isfinite(r["end_to_end_ms"])
            and r["end_to_end_ms"] > 0,
            "invalid cold trace",
        )
    a, b = [[r["end_to_end_ms"] for r in rows if r["arm"] == k] for k in ("serial", "grouped")]
    x, y = median(a), median(b)
    wins = sum(v < u for u, v in zip(a, b, strict=True))
    return dict(
        serial_ms=x,
        grouped_ms=y,
        reduction=1 - y / x,
        wins=wins,
        gate=wins >= 5 and y <= 0.95 * x,
        serial_peak_bytes=max(r["peak_owned_bytes"] for r in rows if r["arm"] == "serial"),
        grouped_peak_bytes=max(r["peak_owned_bytes"] for r in rows if r["arm"] == "grouped"),
    )


def child(kind, cache, dest):
    def die():
        os.kill(os.getpid(), signal.SIGKILL)
        raise RuntimeError("unreachable")

    def on_commit(n):
        if n == 2 and kind == "between":
            die()

    if kind == "inside":
        orig = sr.transfer
        writes = 0

        def intercepted(src, size, h, dest=None, all_bytes=None):
            nonlocal writes
            if isinstance(dest, sqlite3.Blob):
                writes += 1
                if writes == 3:
                    block = src.read(1 << 20)
                    need(len(block) == 1 << 20, "incomplete crash setup")
                    dest.write(block)
                    die()
            return orig(src, size, h, dest, all_bytes)

        sr.transfer = intercepted
    g.issue(cache, dest, committed=on_commit)
    raise RuntimeError("kill not reached")


def killtest(kind, wd, d):
    cache = wd / "cache"
    dst = wd / "unpublished"
    p = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "_kill",
            kind,
            str(cache),
            str(dst),
        ],
        capture_output=True,
        timeout=60,
    )
    need(
        p.returncode == -signal.SIGKILL and not os.path.lexists(dst) and not p.stderr,
        "not an unpublished SIGKILL",
    )
    expected = [s["z"] for s in d["slabs"][:2]]
    # Open once read/write so SQLite can roll its hot journal back before examining the state.
    with closing(sqlite3.connect(cache / "slabs.sqlite3")) as db:
        need(db.execute("pragma integrity_check").fetchall() == [("ok",)], "SQLite integrity")
        rows = db.execute("select z,length(payload) from slabs order by z").fetchall()
        need(
            rows == [(s["z"], sr.size(d, s)) for s in d["slabs"][:2]],
            "partial transaction became visible",
        )
    result = trial("grouped", wd, d, cache, expected)
    return dict(signal=p.returncode, unpublished=True, prefix=rows, resume=result)


def holes(wd, d):
    # Controlled valid but non-prefix checkpoint, populated with independently hashed slices.
    cache = wd / "cache"
    old.validate_cache(cache)
    kept = d["slabs"][::2]
    with (
        sr.writer(cache),
        closing(sqlite3.connect(cache / "slabs.sqlite3")) as db,
        (M / "reference/full-corrected-512.f32").open("rb") as ref,
    ):
        sr.attach(db, g.PLAN, {s["z"]: sr.size(d, s) for s in d["slabs"]})
        for s in kept:
            z = s["z"]
            n = sr.size(d, s)
            ref.seek(z * 4 * d["n"] ** 2)
            with db:
                db.execute("insert into slabs values(?,zeroblob(?))", (z, n))
                with db.blobopen("slabs", "payload", z) as blob:
                    sr.transfer(ref, n, s["sha256"], blob)
    r = trial("grouped", wd, d, cache, [s["z"] for s in kept])
    second = wd / "all-cached"
    second.mkdir(mode=0o700)
    z = trial("grouped", second, d, cache, [s["z"] for s in d["slabs"]])
    return {"non_prefix": r, "fully_cached": z}


def qualify(out):
    out = fresh(out)
    d = sr.plan(g.FIXTURE, g.PLAN)
    need(
        g.sha(M / "reference/full-corrected-512.f32") == d["volume_sha256"],
        "missing verified retained reference",
    )
    timed = []
    warm = []

    def isolated(fn):
        with tempfile.TemporaryDirectory(prefix="grouped-trial-", dir=M / "maintenance") as wd:
            return fn(Path(wd))

    for j, k in [(-1, "serial"), (-1, "grouped"), *balanced()]:
        r = isolated(lambda wd: trial(k, wd, d))
        r["block"] = j
        (warm if j < 0 else timed).append(r)
        print(j, k, round(r["end_to_end_ms"], 1), flush=True)
    chosen = decision(timed, d)
    crashes = {k: isolated(lambda wd: killtest(k, wd, d)) for k in ("between", "inside")}
    gap = isolated(lambda wd: holes(wd, d))
    q = dict(
        format="CFDK-Spark-grouped-replay-qualification-v1",
        source_sha256={
            p.name: g.sha(p)
            for p in [Path(__file__), Path(g.__file__), Path(old.__file__), Path(sr.__file__)]
        },
        plan=g.PLAN,
        warm=warm,
        cold=timed,
        decision=chosen,
        crashes=crashes,
        **gap,
        scope="Sealed measured walnut. Each timed full replay includes source authentication, native work, database commits, assembly, hashing and output fsync; post-run independent offline verification is outside the timer. Fresh caches, warm source pages. Same 768 MiB limit, NOT the same actual peak. Process SIGKILL, not storage power loss. Not independent specimen accuracy.",
    )
    with staging(out, ".grouped-receipt-") as tmp:
        stage = Path(tmp) / "ready"
        stage.mkdir()
        (stage / "qualification.json").write_text(json.dumps(q, indent=2) + "\n")
        publish(stage, out)
    print(json.dumps(chosen))
    print(g.sha(out / "qualification.json"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["_kill"]:
        child(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("out", type=Path)
        qualify(p.parse_args().out)
