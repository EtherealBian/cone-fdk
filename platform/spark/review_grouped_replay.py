"""Independently reopen a pinned grouped-resume certificate using stdlib only, without CUDA or project imports."""

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from statistics import median

PLAN = "bb25b41dae5f7fdccdc58c6692313addafca96973a44c99cc1b97ceec15bd047"
NATIVE = "c1320694248460d2de39f892f377b1b3a7473d186f32d9971d789d45d140f0ec"
ASTRA = "fa5180be58a97e86e4be69c7515a6e8937b98c0a57ba2a3e6b402030a8aa4df4"


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def pairs(items):
    d = dict(items)
    need(len(d) == len(items), "duplicate JSON keys")
    return d


def load(path, pin):
    need(isinstance(pin, str) and re.fullmatch("[a-f0-9]{64}", pin), "external SHA-256 required")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(fd)
        need(stat.S_ISREG(st.st_mode) and 0 < st.st_size <= 2 << 20, "certificate size")
        raw = f.read((2 << 20) + 1)
    need(len(raw) == st.st_size and hashlib.sha256(raw).hexdigest() == pin, "changed certificate")
    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda v: need(False, "nonfinite JSON")
    )


def check(q, d):
    need(
        q["format"] == "CFDK-Spark-grouped-replay-qualification-v1" and q["plan"] == PLAN,
        "wrong certificate",
    )
    zs = [s["z"] for s in d["slabs"]]
    need(d["n"] == 512 and len(zs) == 5 and d["budget_mib"] == 768, "unknown acquisition")
    sizes = {s["z"]: 4 * d["n"] ** 2 * s["depth"] for s in d["slabs"]}
    cap = d["budget_mib"] << 20
    projection = d["projections_bytes"]
    vol = d["volume_sha256"]
    manifest = dict(
        format="CFDK-replayed-volume-v1",
        plan_sha256=PLAN,
        layout="float32 little-endian, z/y/x",
        shape_zyx=[512] * 3,
        volume_bytes=4 * 512**3,
        volume_sha256=vol,
        slabs=d["slabs"],
    )
    mh = hashlib.sha256(
        (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("ascii")
    ).hexdigest()

    def trial(r, arm, keep):
        rest = [z for z in zs if z not in keep]
        num = len(rest) if arm == "serial" else int(bool(rest))
        work = r["report"]
        need(
            r["arm"] == arm
            and r["reused_z"] == keep
            and r["computed_z"] == rest
            and r["manifest_sha256"] == work["manifest_sha256"] == mh
            and r["volume_sha256"] == work["volume_sha256"] == vol
            and work["computed"] == len(rest)
            and work["reused_bytes"] == sum(sizes[z] for z in keep),
            "wrong destination or checkpoints",
        )
        want = {
            "workers": num,
            "filter_batches": 45 * num,
            "projection_bytes": num * projection,
            "bp_launches": 45 * len(rest),
        }
        need(all(type(r[k]) is int and r[k] == v for k, v in want.items()), "wrong work")
        need(
            type(r["end_to_end_ms"]) in (int, float)
            and math.isfinite(r["end_to_end_ms"])
            and r["end_to_end_ms"] > 0,
            "timer",
        )
        peak = r["peak_owned_bytes"]
        need(type(peak) is int and (0 < peak <= cap if num else peak == 0), "memory")
        if arm == "serial":
            nr = work["native"]
            need(
                work["reused"] == len(keep)
                and work["native_projection_bytes_per_invocation"] == projection
                and [v["z"] for v in nr] == rest
                and all(
                    v["gpu"] == "NVIDIA GB10"
                    and v["depth"] * 4 * 512**2 == sizes[v["z"]]
                    and v["driver"] == 13000
                    for v in nr
                )
                and max((v["peak_owned_bytes"] for v in nr), default=0) == peak,
                "wrong serial trace",
            )
        else:
            need(
                work["format"] == "CFDK-GB10-grouped-replay-v1"
                and work["plan"] == PLAN
                and work["parents"] == dict(native=NATIVE, astra=ASTRA)
                and work["source"] == q["source_sha256"]["grouped_replay.py"]
                and work["reused_z"] == keep
                and work["max_group"] == 5
                and len(work["groups"]) == num
                and work["filter_batches"] == want["filter_batches"]
                and work["projection_bytes_read"] == want["projection_bytes"],
                "wrong grouped trace",
            )
            if num:
                group = work["groups"][0]
                n = group["native"]
                need(
                    group["z"] == rest
                    and n["gpu"] == "NVIDIA GB10"
                    and n["redzones_verified"] is True
                    and n["budget_bytes"] == n["effective_budget_bytes"] == cap
                    and n["peak_owned_bytes"] == peak
                    and n["projection_bytes_read"] == projection
                    and n["projection_passes"] == 1
                    and n["filter_batches"] == 45
                    and n["launches_bp"] == 45 * len(rest)
                    and (n["views"], n["batch"], n["n"], n["format"])
                    == (720, 16, 512, "CFDKROI1"),
                    "wrong native counters",
                )
                cur = 20 + 24 * len(rest)
                regions = []
                for z in rest:
                    regions.append(
                        dict(
                            origin_xyz=[0, 0, z],
                            shape_xyz=[512, 512, sizes[z] // (4 * 512**2)],
                            offset_bytes=cur,
                            size_bytes=sizes[z],
                        )
                    )
                    cur += sizes[z]
                need(
                    json.dumps(n["regions"], sort_keys=True)
                    == json.dumps(regions, sort_keys=True)
                    and n["output_bytes"] == cur,
                    "wrong native pack offsets",
                )

    order = [
        (j, k)
        for j in range(6)
        for k in (("serial", "grouped") if j % 2 == 0 else ("grouped", "serial"))
    ]
    rows = q["cold"]
    need(
        [(r["block"], r["arm"]) for r in rows] == order
        and [(r["block"], r["arm"]) for r in q["warm"]] == [(-1, "serial"), (-1, "grouped")],
        "missing or reordered trial",
    )
    for r in q["warm"] + rows:
        trial(r, r["arm"], [])
    times = [
        [r["end_to_end_ms"] for r in rows if r["arm"] == arm] for arm in ("serial", "grouped")
    ]
    x, y = map(median, times)
    wins = sum(v < u for u, v in zip(*times, strict=True))
    res = dict(
        serial_ms=x,
        grouped_ms=y,
        reduction=1 - y / x,
        wins=wins,
        gate=wins >= 5 and y <= 0.95 * x,
        serial_peak_bytes=max(r["peak_owned_bytes"] for r in rows if r["arm"] == "serial"),
        grouped_peak_bytes=max(r["peak_owned_bytes"] for r in rows if r["arm"] == "grouped"),
    )
    need(res == q["decision"], "wrong verdict")
    need(set(q["crashes"]) == {"inside", "between"}, "crash coverage")
    for c in q["crashes"].values():
        need(
            type(c["signal"]) is int
            and c["signal"] == -9
            and c["unpublished"] is True
            and c["prefix"] == [[z, sizes[z]] for z in zs[:2]],
            "partial or published killed delivery",
        )
        trial(c["resume"], "grouped", zs[:2])
    trial(q["non_prefix"], "grouped", zs[::2])
    trial(q["fully_cached"], "grouped", zs)
    return dict(
        decision=res,
        volume_sha256=vol,
        verified_cold_pairs=6,
        verified_kill_scenarios=2,
        nonprefix_missing=[z for z in zs if z not in zs[::2]],
        cached_workers=0,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qualification", type=Path)
    p.add_argument("sha256")
    p.add_argument("plan", type=Path)
    a = p.parse_args()
    print(json.dumps(check(load(a.qualification, a.sha256), load(a.plan, PLAN)), indent=2))
