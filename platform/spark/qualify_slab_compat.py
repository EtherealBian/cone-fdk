"""Interrupt the default worker, then recover through a synchronous legacy CUDA callback."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import stat
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from build_slab_compat import ROOT, build
from sealed_inputs import syncdir

PIN = "bb25b41dae5f7fdccdc58c6692313addafca96973a44c99cc1b97ceec15bd047"
OVERLAY = "f53f88ed79fe450ea502387d6152665ecd453c09a0efbbb76f7940e30dd7769a"
PLAN = ROOT / "fixtures/walnut512-slab-replay-spark-v1.json"


def need(b, msg):
    if not b:
        raise ValueError(msg)


def filehash(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load():
    overlay = ROOT / "build/spark/python-slab-v2"
    build(overlay, OVERLAY)
    sys.path[:0] = [str(overlay), str(ROOT / "scripts")]
    import slab_replay as sr

    need(Path(sr.__file__).resolve() == overlay / "slab_replay.py", "wrong facade")
    return sr


def qualify(scan, products, reference, job, crash=False):
    sr = load()
    d = sr.plan(PLAN, PIN)
    cache = job / "cache"
    out = job / "delivery"
    args = (PLAN, PIN, scan, products, cache, out)
    if crash:

        def stop(count):
            if count == 2:
                os.kill(os.getpid(), signal.SIGKILL)

        sr.generate(*args, committed=stop)
        raise RuntimeError("no crash")
    parent = job.parent.stat()
    need(
        job.absolute() == job.resolve(strict=False)
        and parent.st_uid == os.getuid()
        and stat.S_IMODE(parent.st_mode) & 0o077 == 0,
        "job requires private parent",
    )
    need(filehash(reference) == d["volume_sha256"], "wrong dense reference")
    with (ROOT.parent / "gpu.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job.mkdir(mode=0o700, exist_ok=False)
        cmd = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            *map(str, (scan, products, reference, job)),
            "--crash-child",
        ]
        child = subprocess.run(cmd, capture_output=True, timeout=1200)
        (job / "child.stderr").write_bytes(child.stderr)
        need(
            child.returncode == -signal.SIGKILL and not os.path.lexists(out),
            "worker did not die before delivery",
        )
        with closing(
            sqlite3.connect((cache / "slabs.sqlite3").as_uri() + "?mode=ro", uri=True)
        ) as db:
            rows = db.execute("select z,length(payload) from slabs order by z").fetchall()
            need(
                rows == [(s["z"], sr.size(d, s)) for s in d["slabs"][:2]], "unexpected checkpoint"
            )
            for z, length in rows:
                s = next(s for s in d["slabs"] if s["z"] == z)
                with db.blobopen("slabs", "payload", z, readonly=True) as blob:
                    sr.transfer(blob, length, s["sha256"])
        calls = []

        def worker(sc, build_alias, raw, plan, slab):
            for alias, original in zip(
                sr.paths(sc, build_alias), sr.paths(scan, products), strict=True
            ):
                linked = os.readlink(alias)
                need(
                    linked.startswith(f"/proc/{os.getpid()}/fd/")
                    and not alias.samefile(original),
                    "worker saw original file",
                )
            calls.append(slab["z"])
            return sr.run_native(sc, build_alias, raw, plan, slab)

        result = sr.generate(*args, native=worker)
    need(
        (result["computed"], result["reused"], result["reused_bytes"])
        == (3, 2, sum(n for _, n in rows))
        and calls == [s["z"] for s in d["slabs"][2:]]
        and all(s["gpu"] == "NVIDIA GB10" for s in result["native"]),
        "wrong schedule/device",
    )
    verified = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(ROOT / "scripts/slab_replay.py"),
                "verify",
                str(PLAN),
                str(out),
                "--plan-pin",
                PIN,
            ]
        )
    )
    need(verified == sr.manifest(d, PIN), "detached manifest mismatch")
    size = 0
    with reference.open("rb") as ref, (out / "volume.f32").open("rb") as actual:
        while chunk := ref.read(1 << 20):
            need(actual.read(len(chunk)) == chunk, "different reconstructed bytes")
            size += len(chunk)
        need(not actual.read(1) and size == 4 * d["n"] ** 3, "wrong output length")
    report = dict(
        format="CFDK-GB10-slab-api-recovery-v2",
        overlay_sha256=OVERLAY,
        plan_sha256=PIN,
        qualifier_sha256=filehash(Path(__file__)),
        scope="720 measured walnut views. Same-source process interruption and numerical replay, not a power-loss test or physical accuracy claim.",
        input_geometry_sha256=d["geometry_sha256"],
        input_projection_sha256=d["projections_sha256"],
        signal="SIGKILL",
        exit_code=child.returncode,
        output_present_after_kill=False,
        verified_checkpoint_z_bytes=rows,
        callback_slab_z=calls,
        worker_input="four sealed snapshot paths per synchronous callback, launched in a real child process",
        byte_compared=size,
        compared_voxels=size // 4,
        detached_stdlib_verified=True,
        **result,
    )
    with (job / "qualification.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    syncdir(job)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("scan", "products", "reference", "job"):
        p.add_argument(n, type=Path)
    p.add_argument("--crash-child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    print(
        json.dumps(
            qualify(
                a.scan.absolute(),
                a.products.absolute(),
                a.reference.absolute(),
                a.job.absolute(),
                a.crash_child,
            )
        )
    )
