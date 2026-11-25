"""Detached, stdlib-only review of the pinned GB10 checkpoint-budget experiment. Does not run CUDA."""

import argparse
import functools
import hashlib
import itertools
import json
import math
import os
import re
import stat
from pathlib import Path

PLAN = "bb25b41dae5f7fdccdc58c6692313addafca96973a44c99cc1b97ceec15bd047"
PARENT = "1546c968ae06018b1a9b6b687da760332cbb2613bedc3438a6814a12ea7d7946"
SOURCES = {
    "qualify_budgeted_replay.py": "e8dd27bf8561e358f5298ac13a11ce3df28d1de6fcf8cf5f100cbaf82e6a19dd",
    "budgeted_replay.py": "880d1e21adedee1752c5c04763e412b67b84b0c11aa8d5b3c272e09264e40387",
}
MIB = 1 << 20


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def enc(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)


def eq(a, b):
    need(enc(a) == enc(b), "mismatch in " + str(b)[:96])


def load(path, pin):
    need(type(pin) is str and re.fullmatch("[a-f0-9]{64}", pin), "external SHA-256 required")

    def pairs(v):
        d = dict(v)
        need(len(d) == len(v), "duplicate key")
        return d

    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as f:
        s = os.fstat(f.fileno())
        need(stat.S_ISREG(s.st_mode) and 0 < s.st_size <= 2 * MIB, "bad file size or type")
        raw = f.read(2 * MIB + 1)
    need(
        len(raw) == s.st_size and hashlib.sha256(raw).hexdigest() == pin, "external hash mismatch"
    )
    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda v: need(False, "nonfinite number")
    )


class Model:
    def __init__(self, d, base):
        self.z = tuple(s["z"] for s in d["slabs"])
        self.vol = {s["z"]: 4 * d["n"] ** 2 * s["depth"] for s in d["slabs"]}
        self.base = base
        self.proj = d["projections_bytes"]
        # Independent enumeration: partitions are equal-label classes of restricted growth strings.
        # All tuples are generated without using fit/budget or the production adding-to-bins algorithm.
        self.parts = {}
        for mask in range(32):
            rest = tuple(z for i, z in enumerate(self.z) if mask >> i & 1)
            p = []
            for labs in itertools.product(range(len(rest)), repeat=len(rest)):
                if any(x > 1 + max(labs[:i], default=-1) for i, x in enumerate(labs)):
                    continue
                groups = tuple(
                    tuple(z for z, l in zip(rest, labs) if l == i)
                    for i in range(max(labs, default=-1) + 1)
                )
                p.append((len(groups), max(map(self.peak, groups), default=0), groups))
            self.parts[rest] = sorted(p)

    def peak(self, g):
        return self.base + sum(self.vol[z] for z in g)

    def legal(self, rest, cap):
        return [c for c in self.parts[rest] if not c[0] or c[1] + 512 <= cap * MIB]

    def schedule(self, present, cap):
        rest = tuple(z for z in self.z if z not in present)
        v = self.legal(rest, cap)
        num, mx, gs = v[0]
        return dict(
            format="CFDK-GB10-budget-schedule-v1",
            plan=PLAN,
            profile=PARENT,
            cap_mib=cap,
            reused_z=sorted(present),
            pending_z=list(rest),
            jobs=num,
            peak_owned_bytes=mx,
            projected_projection_bytes=num * self.proj,
            feasible_partitions=len(v),
            groups=[
                dict(
                    z=list(g), peak_owned_bytes=self.peak(g), planner_min_bytes=self.peak(g) + 512
                )
                for g in gs
            ],
        )

    @functools.cached_property
    def census(self):
        accepted = refused = 0
        front = []
        h = hashlib.sha256()
        for mask in range(32):
            rest = tuple(z for i, z in enumerate(self.z) if mask >> i & 1)
            prev = None
            for cap in range(1, 769):
                v = self.legal(rest, cap)
                if not v:
                    refused += 1
                    continue
                jobs, mx, gs = v[0]
                accepted += 1
                flat = [z for g in gs for z in g]
                h.update((enc([mask, cap, [jobs, mx], flat]) + "\n").encode())
                if mask == 31 and prev != (jobs, mx):
                    front.append(dict(cap_mib=cap, workers=jobs, peak_bytes=mx))
                    prev = (jobs, mx)
        return dict(
            cases=accepted + refused,
            accepted=accepted,
            refused=refused,
            sha256=h.hexdigest(),
            frontier=front,
        )


def timer(t):
    need(type(t) in (int, float) and math.isfinite(t) and t > 0, "bad timer")


def check(q, d, parent):
    eq(q["format"], "CFDK-GB10-budget-frontier-qualification-v1")
    eq(q["plan"], PLAN)
    eq(q["parent"], PARENT)
    eq(q["source_sha256"], SOURCES)
    eq(parent["plan"], PLAN)
    eq(parent["decision"]["gate"], True)
    eq((d["n"], d["batch"], len(d["slabs"])), (512, 16, 5))
    ref = next(r for r in parent["cold"] if r["arm"] == "grouped")
    vol = d["volume_sha256"]
    eq(ref["volume_sha256"], vol)
    base = ref["report"]["groups"][0]["native"]["peak_owned_bytes"] - 4 * d["n"] ** 3
    m = Model(d, base)
    eq(q["independent"], m.census)
    front = m.census["frontier"]
    caps = [f["cap_mib"] for f in front]
    eq(caps, [161, 217, 253, 326, 555])
    eq(q["refused_cold_mib"], caps[0] - 1)
    eq(q["refusal"], "cap cannot hold the largest missing slab and filter workspace at batch 16")
    eq(len(q["runs"]), len(front))
    native_workers = 0
    delivery_count = 0
    for trial, steps in zip(
        q["runs"] + [q["changed_caps"]],
        [[(c, 5)] for c in caps] + [[(253, 1), (161, 1), (326, 1), (1, 1)]],
        strict=True,
    ):
        eq(trial["steps"], steps)
        eq(len(trial["phases"]), len(steps))
        saved = []
        for row, (cap, limit) in zip(trial["phases"], steps, strict=True):
            eq(row["format"], "CFDK-GB10-budget-warm-v1")
            eq(row["source_sha256"], SOURCES["budgeted_replay.py"])
            eq(row["issued_profile"], PARENT)
            eq(row["original_plan_sha256"], PLAN)
            eq(row["max_jobs"], limit)
            timer(row["prepublication_ms"])
            sched = m.schedule(saved, cap)
            eq(row["schedule"], sched)
            issued = sched["groups"][:limit]
            eq(len(row["issued"]), len(issued))
            mx = 0
            for rr, gg in zip(row["issued"], issued, strict=True):
                n = rr["native"]
                gs = gg["z"]
                eq(rr["z"], gs)
                timer(rr["process_ms"])
                native_workers += 1
                mx = max(mx, gg["peak_owned_bytes"])
                cur = 20 + 24 * len(gs)
                regions = []
                for z in gs:
                    regions.append(
                        dict(
                            origin_xyz=[0, 0, z],
                            shape_xyz=[512, 512, m.vol[z] // (4 * 512**2)],
                            size_bytes=m.vol[z],
                            offset_bytes=cur,
                        )
                    )
                    cur += m.vol[z]
                    saved.append(z)
                expect = dict(
                    gpu="NVIDIA GB10",
                    driver_api=13000,
                    cufft_version=12000,
                    views=720,
                    n=512,
                    batch=16,
                    requested_batch=16,
                    planner="batch-first",
                    voxel_spacing_mm=d["fov_mm"] / d["n"],
                    format="CFDKROI1",
                    budget_bytes=cap * MIB,
                    effective_budget_bytes=cap * MIB,
                    redzones_verified=True,
                    projection_passes=1,
                    projection_bytes_read=m.proj,
                    filter_batches=45,
                    launches_bp=45 * len(gs),
                    feasible_plans_scored=[dict(batch=16, passes=1)],
                    regions=regions,
                    output_bytes=cur,
                    peak_owned_bytes=gg["peak_owned_bytes"],
                    fft_workspace_bytes=0,
                )
                for k, v in expect.items():
                    eq(n[k], v)
                for k in ("host_pinned_bytes", "free_before_bytes", "free_at_allocation_bytes"):
                    need(type(n[k]) is int and n[k] > 0, "allocation record")
                for k in ("filter_ms", "bp_ms", "h2d_ms", "d2h_ms", "wall_ms"):
                    timer(n[k])
            eq(row["actual_peak_owned_bytes"], mx)
            eq(row["actual_projection_bytes"], len(issued) * m.proj)
            eq(row["present_z"], sorted(saved))
            eq(row["remaining"], m.schedule(saved, cap))
        eq(sorted(saved), list(m.z))
        eq(trial["final_workers"], 0)
        eq(trial["volume"], vol)
        eq(trial["manifest"], ref["manifest_sha256"])
        delivery_count += 1
    return dict(
        cold_frontier=front,
        grid_cases=m.census["cases"],
        feasible=m.census["accepted"],
        refused=m.census["refused"],
        verified_native_workers=native_workers,
        verified_deliveries=delivery_count,
        volume_sha256=vol,
        changed_caps_mib=[253, 161, 326, 1],
        final_cached_workers=0,
        scope="Offline consistency of pinned logged runs and independent finite scheduling search; not a new CUDA run or a proof of physical dimensions.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qualification", type=Path)
    p.add_argument("sha256")
    p.add_argument("plan", type=Path)
    p.add_argument("grouped_parent", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            check(
                load(a.qualification, a.sha256),
                load(a.plan, PLAN),
                load(a.grouped_parent, PARENT),
            ),
            indent=2,
        )
    )
