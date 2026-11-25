"""Exercise a real mid-volume interruption on measured projections against a frozen dense baseline."""

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from slab_replay import STEP, encode, generate, manifest, need, plan, size

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/slab_replay.py"
PLAN = ROOT / "fixtures/walnut512-slab-replay-v1.json"
PLAN_PIN = "0c8da08152e13be98a418810f7832c6fc873ea7cc546de8ba8275a09715e50ca"
RULES = "17848db29197b7dcba805541cc8809553210d0318f00a27de74c3654bf898c18"
REFERENCE = {
    "native-ab.json": "ec2657537c64e115a5b6273bb84ad81b849405f208a1762c262d85a40be436d1",
    "run-repeat.json": "f594d508c95e96565f83c92715ad6a1d23d16572be297b6b052c17ede0c78ecd",
}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def entry(args):
    def kill_after_second(n):
        if n == 2:
            os._exit(
                73
            )  # No Python unwinding, closing the SQLite lease, or publishing a final directory.

    generate(PLAN, PLAN_PIN, *map(Path, args), committed=kill_after_second)
    raise RuntimeError("hard-exit case failed to interrupt")


def child(args, expected=0):
    p = subprocess.run([sys.executable, *map(str, args)], capture_output=True, timeout=1200)
    need(
        p.returncode == expected,
        "unexpected child exit: " + p.stderr.decode("utf8", errors="replace")[-1500:],
    )
    return json.loads(p.stdout) if expected == 0 else p.returncode


def compare_outputs(original, deliveries, d):
    """Direct ordered byte comparison to the original dense run, separately from the recipe verifier."""
    from contextlib import ExitStack

    planes = []
    with ExitStack() as stack:
        files = [
            stack.enter_context(p.open("rb"))
            for p in [original, *(v / "volume.f32" for v in deliveries)]
        ]
        for s in d["slabs"]:
            h = hashlib.sha256()
            remaining = size(d, s)
            while remaining:
                n = min(STEP, remaining)
                b = files[0].read(n)
                need(
                    len(b) == n and all(f.read(n) == b for f in files[1:]),
                    "dense vs checkpoint bytes differ",
                )
                h.update(b)
                remaining -= n
            need(
                h.hexdigest() == s["sha256"], "baseline slab no longer equals precommitted slice"
            )
            planes.append(
                dict(z=s["z"], depth=s["depth"], bytes=size(d, s), sha256=h.hexdigest())
            )
        need(all(not f.read(1) for f in files), "trailing baseline or delivery")
    return planes


def qualify(scan, build, dense, work, archive):
    scan, build, dense, work, archive = (
        Path(p).resolve() for p in (scan, build, dense, work, archive)
    )
    need(
        sha(ROOT / "fixtures/slab-replay-protocol-v1.json") == RULES, "changed recovery protocol"
    )
    d = plan(PLAN, PLAN_PIN)
    for n, x in REFERENCE.items():
        need(sha(archive / n) == x, "changed historical report")
    a = json.loads((archive / "native-ab.json").read_bytes())["cases"]["full-corrected"]
    v = json.loads((archive / "run-repeat.json").read_bytes())["baseline_outputs"]["walnut512"]
    need(
        a["geom_sha256"] == d["geometry_sha256"]
        and a["projections_sha256"] == d["projections_sha256"]
        and sha(dense) == d["volume_sha256"] == v,
        "wrong dense baseline or acquisition",
    )
    work.mkdir(parents=True, exist_ok=False)
    resumecache = work / "resume-cache"
    resumed = work / "resumed"
    cold = work / "cold"
    coldcache = work / "cold-cache"
    code = child([Path(__file__), "_interrupt", scan, build, resumecache, resumed], expected=73)
    need(not resumed.exists(), "early published volume")
    with closing(sqlite3.connect(resumecache / "slabs.sqlite3")) as db:
        committed = db.execute("select z, length(payload) from slabs order by z").fetchall()
    need(
        committed == [(s["z"], size(d, s)) for s in d["slabs"][:2]],
        "wrong checkpoint after hard exit",
    )

    def replay(cache, dest):
        return child([WORKER, "run", PLAN, scan, build, cache, dest, "--plan-pin", PLAN_PIN])

    r = replay(resumecache, resumed)
    c = replay(coldcache, cold)
    need(
        (r["reused"], r["computed"], c["reused"], c["computed"]) == (2, 3, 0, 5)
        and r["reused_bytes"] == sum(v for _, v in committed)
        and c["reused_bytes"] == 0,
        "wrong resume or cold execution counts",
    )
    need(
        [m["z"] for m in r["native"]] == [s["z"] for s in d["slabs"][2:]]
        and [m["z"] for m in c["native"]] == [s["z"] for s in d["slabs"]],
        "unexpected native jobs",
    )
    signed = manifest(d, PLAN_PIN)
    need(
        r["manifest_sha256"]
        == c["manifest_sha256"]
        == sha(resumed / "manifest.json")
        == sha(cold / "manifest.json")
        and r["volume_sha256"] == c["volume_sha256"] == v,
        "not byte-identical manifests or volumes",
    )
    for dest in (resumed, cold):
        need(
            child(["-I", "-S", WORKER, "verify", PLAN, dest, "--plan-pin", PLAN_PIN]) == signed,
            "portable volume check",
        )
    slabs = compare_outputs(dense, [resumed, cold], d)
    result = dict(
        format="CFDK-slab-replay-qualification-v1",
        scope="Real 720-view corrected Helsinki walnut. Exact reconstruction replay only. Not a new physical accuracy, power-loss or throughput qualification.",
        protocol_sha256=RULES,
        plan_sha256=PLAN_PIN,
        historical_reports=REFERENCE,
        source_sha256={
            n: sha(ROOT / "scripts" / n)
            for n in ("slab_replay.py", "qualify_slab_replay.py", "common.py")
        },
        native_products=d["products"],
        directly_compared_voxels=d["n"] ** 3,
        original_volume_sha256=v,
        interrupted=dict(
            exit_code=code, complete_output_present=False, committed_z_and_bytes=committed
        ),
        resumed=r,
        cold=c,
        slabs_compared=slabs,
        cache_bytes={
            tag: (folder / "slabs.sqlite3").stat().st_size
            for tag, folder in [("cold", coldcache), ("resumed", resumecache)]
        },
        limits="Each new slab invokes the original native CLI on all 720 views; logical input file is 238694400 bytes. No claim on OS cache hits, disk traffic or speed. Reuse requires trusted per-slab and dense baseline digests.",
    )
    (work / "qualification.json").write_bytes(encode(result))
    print(
        "PASS: exited after 2, reused 2/computed 3; cold computed 5; both complete volumes match the frozen measured-data reconstruction"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_interrupt":
        entry(sys.argv[2:])
    else:
        p = argparse.ArgumentParser(description=__doc__)
        for n in ("scan", "build", "dense", "work", "archive"):
            p.add_argument(n, type=Path)
        qualify(**vars(p.parse_args()))
