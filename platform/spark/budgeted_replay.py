"""Prepare verified cache rows under an explicit GB10 per-worker memory limit, even across sparse resumes."""

import argparse
import fcntl
import json
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path

import grouped_replay as g
import review_grouped_replay as review
from grouped_replay import M, MIB, need, sr, old
from spark_publication import fresh, publish, staging

CERT = "1546c968ae06018b1a9b6b687da760332cbb2613bedc3438a6814a12ea7d7946"
# Fixed authenticated walnut geometry and stable batch=16. Actual checked cuFFT workspace = 0.
# 5 non-empty guarded device buffers; native planner also reserves a guard for the empty work buffer.
LIVE = 16 * 296 * 1024 * 8 + 16 * 296 * 280 * 4 + 1024 * 4 + 720 * 8 + 5 * 512
RESERVE = 512


def evidence():
    d = review.load(g.FIXTURE, g.PLAN)
    c = review.load(M / "grouped-replay-qualified-001/qualification.json", CERT)
    v = review.check(c, d)
    need(
        v["decision"]["gate"] is True
        and c["source_sha256"]["grouped_replay.py"] == g.sha(g.__file__),
        "unqualified native envelope",
    )
    for r in c["cold"]:
        if r["arm"] == "grouped":
            n = r["report"]["groups"][0]["native"]
            need(
                n["fft_workspace_bytes"] == 0 and n["peak_owned_bytes"] == LIVE + 4 * d["n"] ** 3,
                "workspace changed",
            )
    return d


def make_plan(d, saved, cap_mib):
    need(type(cap_mib) is int and 1 <= cap_mib <= 768, "cap must be an integer from 1 to 768 MiB")
    zs = [s["z"] for s in d["slabs"]]
    need(
        len(zs) == 5
        and len(set(zs)) == 5
        and isinstance(saved, (list, tuple, dict, set, frozenset))
        and all(type(z) is int and z in zs for z in saved)
        and len(saved) == len(set(saved)),
        "wrong cache keys",
    )
    size = {s["z"]: sr.size(d, s) for s in d["slabs"]}
    rest = [z for z in zs if z not in saved]
    cap = cap_mib * MIB

    def peak(group):
        return LIVE + sum(size[z] for z in group)

    # At most Bell(5)=52 unordered partitions. Enumerate once in canonical first-occurrence order.
    valid = []

    def visit(i, groups):
        if i == len(rest):
            groups = tuple(tuple(x) for x in groups)
            valid.append((len(groups), max(map(peak, groups), default=0), groups))
            return
        z = rest[i]
        for j in range(len(groups)):
            if peak(groups[j]) + size[z] + RESERVE <= cap:
                groups[j].append(z)
                visit(i + 1, groups)
                groups[j].pop()
        if peak([z]) + RESERVE <= cap:
            groups.append([z])
            visit(i + 1, groups)
            groups.pop()

    visit(0, [])
    need(valid, "cap cannot hold the largest missing slab and filter workspace at batch 16")
    jobs, maxlive, groups = min(valid)
    return dict(
        format="CFDK-GB10-budget-schedule-v1",
        plan=g.PLAN,
        profile=CERT,
        cap_mib=cap_mib,
        reused_z=sorted(saved),
        pending_z=rest,
        jobs=jobs,
        peak_owned_bytes=maxlive,
        projected_projection_bytes=jobs * d["projections_bytes"],
        feasible_partitions=len(valid),
        groups=[
            dict(
                z=list(group),
                peak_owned_bytes=peak(group),
                planner_min_bytes=peak(group) + RESERVE,
            )
            for group in groups
        ],
    )


def warm(cache, out, cap_mib, max_jobs=5):
    start = time.monotonic_ns()
    need(type(max_jobs) is int and 1 <= max_jobs <= 5, "job allowance must be 1..5")
    need(".." not in Path(cache).parts, "cache traversal")
    out = old.private_location(fresh(out))
    cache = old.private_location(cache)
    need(not cache.is_relative_to(out) and not out.is_relative_to(cache), "overlapping output")
    d = evidence()
    need(
        all(
            not p.resolve().is_relative_to(loc) and not loc.is_relative_to(p.resolve())
            for p in (g.FIXTURE, *sr.paths(g.SCAN, g.BUILD))
            for loc in (cache, out)
        ),
        "input alias",
    )
    av = old.sr.plan(g.FIXTURE, g.PLAN)
    need(av == d, "baseline schema")
    if not cache.exists():
        make_plan(d, set(), cap_mib)
    sized = dict(d, budget_mib=cap_mib)
    byz = {s["z"]: s for s in d["slabs"]}
    issued = []
    with old.sealed_scan(g.SCAN, g.BUILD, d) as fds:
        old.validate_cache(cache)
        with sr.writer(cache), closing(sqlite3.connect(cache / "slabs.sqlite3", timeout=0)) as db:
            saved = sr.attach(db, g.PLAN, {z: sr.size(d, s) for z, s in byz.items()})
            schedule = make_plan(d, saved, cap_mib)
            for z in sorted(saved):
                with db.blobopen("slabs", "payload", z, readonly=True) as blob:
                    sr.transfer(blob, sr.size(d, byz[z]), byz[z]["sha256"])
            for group in schedule["groups"][:max_jobs]:
                slabs = [byz[z] for z in group["z"]]
                with tempfile.TemporaryDirectory(dir=cache, prefix=".compute-") as td:
                    raw = Path(td) / "pack"
                    r, ms = g.native(fds, raw, sized, slabs)
                    need(
                        type(r["fft_workspace_bytes"]) is int
                        and r["fft_workspace_bytes"] == 0
                        and r["peak_owned_bytes"] == group["peak_owned_bytes"]
                        and r["effective_budget_bytes"] == cap_mib * MIB
                        and json.dumps(r["feasible_plans_scored"], sort_keys=True)
                        == json.dumps([{"batch": 16, "passes": 1}], sort_keys=True),
                        "native resource model changed",
                    )
                    g.fill(db, sized, slabs, raw, r, None, 0)
                issued.append(dict(z=group["z"], native=r, process_ms=ms))
            present = [z for (z,) in db.execute("select z from slabs order by z")]
            need(
                present == sorted(list(saved) + [z for r in issued for z in r["z"]]),
                "unexpected checkpoint after work",
            )
            remaining = make_plan(d, present, cap_mib)
            rec = dict(
                format="CFDK-GB10-budget-warm-v1",
                source_sha256=g.sha(__file__),
                original_plan_sha256=g.PLAN,
                issued_profile=CERT,
                schedule=schedule,
                max_jobs=max_jobs,
                issued=issued,
                present_z=present,
                remaining=remaining,
                actual_projection_bytes=sum(r["native"]["projection_bytes_read"] for r in issued),
                actual_peak_owned_bytes=max(
                    (r["native"]["peak_owned_bytes"] for r in issued), default=0
                ),
                prepublication_ms=(time.monotonic_ns() - start) / 1e6,
                scope="Durable checkpoint warm-up only; not an assembled delivery. The cap is the per-worker device buffer allocation including guards and FFT work, not host buffers, runtime/driver, RSS or total unified memory. Receipt is point in time; a subsequent command independently revalidates cache bytes.",
            )
            with staging(out, ".budget-warm-") as tmp:
                stage = Path(tmp) / "ready"
                stage.mkdir()
                (stage / "warm.json").write_text(json.dumps(rec, indent=2) + "\n")
                publish(stage, out)
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser(
        "plan",
        help="dry-run from externally supplied saved offsets; does not certify cached bytes",
    )
    plan.add_argument("--saved-z", nargs="*", type=int, default=[])
    run = sub.add_parser("warm")
    run.add_argument("cache", type=Path)
    run.add_argument("out", type=Path)
    run.add_argument("--max-jobs", type=int, default=5)
    for arg in (plan, run):
        arg.add_argument("--cap-mib", required=True, type=int)
    a = p.parse_args()
    if a.cmd == "plan":
        need(len(a.saved_z) == len(set(a.saved_z)), "duplicate saved offset")
        print(json.dumps(make_plan(evidence(), a.saved_z, a.cap_mib), indent=2))
    else:
        with (M / "gpu.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            q = warm(a.cache, a.out, a.cap_mib, a.max_jobs)
            print(
                json.dumps(
                    {
                        k: q[k]
                        for k in (
                            "present_z",
                            "actual_projection_bytes",
                            "actual_peak_owned_bytes",
                        )
                    }
                )
            )
