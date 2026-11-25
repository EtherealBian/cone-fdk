"""Filter a view batch once for several missing slabs; preserve the original SQLite checkpoints and delivery."""

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import struct
import subprocess
import tempfile
import time
from contextlib import closing
from pathlib import Path

import replay_spark as old
from replay_spark import sr
from sealed_inputs import snapshot, syncdir
from spark_publication import fresh, staging, publish
from tune_backproject import PLAN, NATIVE, ASTRA, ROOT, MIB, json_checked, sha
from residual_witness import need

M = ROOT.parent
FIXTURE = ROOT / "fixtures/walnut512-slab-replay-spark-v1.json"
SCAN = M / "reference/full-corrected"
BUILD = ROOT / "build"


def header(plan, slabs):
    n = plan["n"]
    return struct.pack("<8sIfI", b"CFDKROI1", n, plan["fov_mm"] / n, len(slabs)) + b"".join(
        struct.pack("<6I", 0, 0, s["z"], n, n, s["depth"]) for s in slabs
    )


def schedule(plan, saved, max_group):
    need(type(max_group) is int and 1 <= max_group <= 5, "group size must be 1..5")
    missing = [s for s in plan["slabs"] if s["z"] not in saved]
    return [missing[i : i + max_group] for i in range(0, len(missing), max_group)]


def check_report(r, d, slabs):
    n = d["n"]
    off = len(header(d, slabs))
    regions = []
    for s in slabs:
        size = sr.size(d, s)
        regions.append(
            dict(
                origin_xyz=[0, 0, s["z"]],
                shape_xyz=[n, n, s["depth"]],
                offset_bytes=off,
                size_bytes=size,
            )
        )
        off += size
    keys = dict(
        format="CFDKROI1",
        gpu="NVIDIA GB10",
        driver_api=13000,
        cufft_version=12000,
        n=n,
        views=720,
        batch=16,
        requested_batch=16,
        planner="batch-first",
        projection_passes=1,
        filter_batches=45,
        launches_bp=45 * len(slabs),
        projection_bytes_read=d["projections_bytes"],
        budget_bytes=d["budget_mib"] << 20,
        effective_budget_bytes=d["budget_mib"] << 20,
        voxel_spacing_mm=d["fov_mm"] / n,
        redzones_verified=True,
        regions=regions,
        output_bytes=off,
    )
    # JSON booleans cannot impersonate an integer in an overlapping nested field.
    need(
        all(
            k in r
            and type(r[k]) is type(v)
            and json.dumps(r[k], sort_keys=True) == json.dumps(v, sort_keys=True)
            for k, v in keys.items()
        ),
        "incorrect packed coverage, tile layout or counters",
    )
    need(
        type(r["peak_owned_bytes"]) is int
        and 0 < r["peak_owned_bytes"] <= r["budget_bytes"]
        and r["host_pinned_bytes"] > 0,
        "wrong buffers",
    )
    return regions


def native(fds, out, plan, slabs):
    geom, pr, exe, ptx = (f"/proc/self/fd/{fd}" for fd in fds)
    n = plan["n"]
    raw = "".join(f"0,0,{s['z']},{n},{n},{s['depth']}\n" for s in slabs).encode()
    listing = out.parent / "boxes.txt"
    listing.write_bytes(raw)
    with snapshot(listing, hashlib.sha256(raw).hexdigest(), limit=4096, exact=len(raw)) as box:
        start = time.monotonic_ns()
        p = subprocess.run(
            [
                exe,
                "--ptx",
                ptx,
                "--geom",
                geom,
                "--input",
                pr,
                "--output",
                str(out),
                "--n",
                str(n),
                "--fov",
                str(plan["fov_mm"]),
                "--batch",
                str(plan["batch"]),
                "--budget",
                str(plan["budget_mib"]),
                "--roi-list",
                f"/proc/self/fd/{box}",
            ],
            pass_fds=(*fds, box),
            capture_output=True,
            timeout=180,
        )
        dt = (time.monotonic_ns() - start) / 1e6
    need(
        not p.returncode and not p.stderr,
        "packed worker: " + p.stderr.decode(errors="replace")[-700:],
    )
    return json.loads(p.stdout), dt


def validate_range_file(f, d, slabs, regions):
    need(f.read(len(header(d, slabs))) == header(d, slabs), "wrong ROI pack header")
    for s, reg in zip(slabs, regions, strict=True):
        need(f.tell() == reg["offset_bytes"], "wrong region offset")
        sr.transfer(f, reg["size_bytes"], s["sha256"])
    need(not f.read(1), "trailing pack bytes")


def fill(db, d, slabs, raw, r, committed, previous):
    regions = check_report(r, d, slabs)
    fd = os.open(raw, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        info = os.fstat(fd)
        need(
            stat.S_ISREG(info.st_mode)
            and info.st_nlink == 1
            and info.st_size == r["output_bytes"],
            "pack type or size",
        )
        validate_range_file(f, d, slabs, regions)
        # The worker is finished and every payload matches its own externally pinned range.
        # Unname the private transient before durable commits (SIGKILL then closes the read fd).
        need(os.path.samestat(info, raw.lstat()), "pack identity changed")
        raw.unlink()
        for i, (s, reg) in enumerate(zip(slabs, regions, strict=True)):
            f.seek(reg["offset_bytes"])
            z = s["z"]
            size = reg["size_bytes"]
            with db:
                db.execute("INSERT INTO slabs VALUES(?,zeroblob(?))", (z, size))
                with db.blobopen("slabs", "payload", z) as blob:
                    sr.transfer(f, size, s["sha256"], blob)
            if committed:
                committed(previous + i + 1)


def issue(cache, out, *, max_group=5, runner=native, committed=None):
    started = time.monotonic_ns()
    out = fresh(out)
    cache = old.private_location(cache)
    out = old.private_location(out)
    d = sr.plan(FIXTURE, PLAN)
    a = json_checked(M / "walnut-astra-gb10-001/result.json", ASTRA)
    n = json_checked(M / "walnut-gb10-001/result.json", NATIVE)
    need(
        a["passed"] is True
        and n["input_sha256"] == a["input_sha256"]
        and n["output_sha256"] == [d["volume_sha256"]] * 2,
        "unqualified acquisition",
    )
    schedule(d, set(), max_group)
    need(
        not cache.is_relative_to(out)
        and not out.is_relative_to(cache)
        and all(
            not p.resolve().is_relative_to(loc) and not loc.is_relative_to(p.resolve())
            for p in (FIXTURE, *sr.paths(SCAN, BUILD))
            for loc in (cache, out)
        ),
        "cache/output overlaps source or other destination",
    )
    groups = []
    reused_bytes = 0
    computed = 0
    with old.sealed_scan(SCAN, BUILD, d) as fds:
        old.validate_cache(cache)
        with sr.writer(cache), closing(sqlite3.connect(cache / "slabs.sqlite3", timeout=0)) as db:
            saved = sr.attach(db, PLAN, {s["z"]: sr.size(d, s) for s in d["slabs"]})
            missing = schedule(d, saved, max_group)
            # Recheck every reused BLOB *before* allowing any new GPU work.
            for s in d["slabs"]:
                if s["z"] in saved:
                    with db.blobopen("slabs", "payload", s["z"], readonly=True) as blob:
                        sr.transfer(blob, sr.size(d, s), s["sha256"])
                        reused_bytes += sr.size(d, s)
            for slabs in missing:
                with tempfile.TemporaryDirectory(dir=cache, prefix=".compute-") as tmp:
                    raw = Path(tmp) / "out.cfdk"
                    r, dt = runner(fds, raw, d, slabs)
                    fill(db, d, slabs, raw, r, committed, computed)
                groups.append(dict(z=[s["z"] for s in slabs], native=r, process_ms=dt))
                computed += len(slabs)
            with staging(out, ".assemble-") as tmp:
                stage = Path(tmp) / "ready"
                stage.mkdir(mode=0o700)
                whole = hashlib.sha256()
                with (stage / "volume.f32").open("xb") as f:
                    for s in d["slabs"]:
                        with db.blobopen("slabs", "payload", s["z"], readonly=True) as blob:
                            sr.transfer(blob, sr.size(d, s), s["sha256"], f, whole)
                need(whole.hexdigest() == d["volume_sha256"], "assembled SHA mismatch")
                manifest = sr.encode(sr.manifest(d, PLAN))
                (stage / "manifest.json").write_bytes(manifest)
                publish(stage, out)
    return dict(
        format="CFDK-GB10-grouped-replay-v1",
        plan=PLAN,
        parents=dict(native=NATIVE, astra=ASTRA),
        source=sha(__file__),
        max_group=max_group,
        reused_z=sorted(saved),
        computed=computed,
        reused_bytes=reused_bytes,
        groups=groups,
        projection_bytes_read=sum(r["native"]["projection_bytes_read"] for r in groups),
        filter_batches=sum(r["native"]["filter_batches"] for r in groups),
        volume_sha256=d["volume_sha256"],
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        verified_wall_ms=(time.monotonic_ns() - started) / 1e6,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--max-group", type=int, default=5)
    a = p.parse_args()
    with (M / "gpu.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(json.dumps(issue(a.cache, a.out, max_group=a.max_group)))
