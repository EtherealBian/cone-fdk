"""Cross-check every cap/subset with an independent subset DP and execute each cold frontier on CUDA."""

import argparse
import functools
import hashlib
import itertools
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
import budgeted_replay as b
import grouped_replay as g
from grouped_replay import need, M, sr
from qualify_grouped_replay import offline
from spark_publication import fresh, publish, staging

CAPS = (161, 217, 253, 326, 555)


def oracle(d):
    # No production scheduler or LIVE constant: infer the non-volume intercept from the pinned run,
    # then compute min max live bytes for each number of groups by canonical subset DP.
    c = b.review.load(M / "grouped-replay-qualified-001/qualification.json", b.CERT)
    root = c["decision"]["grouped_peak_bytes"] - d["n"] ** 3 * 4
    sizes = [sr.size(d, s) for s in d["slabs"]]
    z = [s["z"] for s in d["slabs"]]

    @functools.lru_cache(None)
    def peaks(mask):
        if not mask:
            return {0: 0}
        ans = {}
        sub = mask
        while sub:
            if sub & (mask & -mask):
                live = root + sum(size for i, size in enumerate(sizes) if sub >> i & 1)
                for k, v in peaks(mask ^ sub).items():
                    ans[k + 1] = min(ans.get(k + 1, 10**16), max(v, live))
            sub = (sub - 1) & mask
        return ans

    h = hashlib.sha256()
    tested = accepted = blocked = 0
    crossovers = []
    for mask in range(32):
        saved = [v for i, v in enumerate(z) if not mask >> i & 1]
        expect = peaks(mask)
        prev = None
        for cap in range(1, 769):
            # Native planner's synthetic empty-work guard is 2 x 256 bytes; not an owned allocation.
            good = [(cnt, v) for cnt, v in expect.items() if not cnt or v + 512 <= cap * 2**20]
            tested += 1
            if not good:
                try:
                    b.make_plan(d, saved, cap)
                except ValueError as e:
                    need("largest missing slab" in str(e), "incorrect refusal")
                else:
                    raise ValueError("unfit accepted")
                blocked += 1
                continue
            ans = b.make_plan(d, saved, cap)
            want = min(good)
            need((ans["jobs"], ans["peak_owned_bytes"]) == want, "not Pareto optimal")
            flat = [x for group in ans["groups"] for x in group["z"]]
            need(
                sorted(flat) == [v for v in z if v not in saved]
                and all(x["planner_min_bytes"] <= cap * 2**20 for x in ans["groups"]),
                "incomplete plan",
            )
            accepted += 1
            h.update((json.dumps((mask, cap, want, flat), separators=(",", ":")) + "\n").encode())
            if mask == 31 and prev != want:
                crossovers.append(dict(cap_mib=cap, workers=want[0], peak_bytes=want[1]))
                prev = want
    need([v["cap_mib"] for v in crossovers] == list(CAPS), "changed physical frontier")
    return dict(
        cases=tested,
        accepted=accepted,
        refused=blocked,
        sha256=h.hexdigest(),
        frontier=crossovers,
    )


def closed_cache(cache, d):
    with closing(sqlite3.connect(f"file:{cache}/slabs.sqlite3?mode=ro", uri=True)) as db:
        for s in d["slabs"]:
            with db.blobopen("slabs", "payload", s["z"], readonly=True) as blob:
                sr.transfer(blob, sr.size(d, s), s["sha256"])


def real(wd, d, steps):
    wd = Path(wd)
    cache = wd / "cache"
    rows = []
    ordered = []
    for i, (cap, limit) in enumerate(steps):
        dest = wd / f"phase-{i}"
        r = b.warm(cache, dest, cap, limit)
        need(
            r == json.loads((dest / "warm.json").read_text())
            and r["actual_peak_owned_bytes"] <= cap * 2**20,
            "warm receipt or cap",
        )
        need(r["schedule"]["reused_z"] == sorted(ordered), "did not authenticate expected cache")
        for q in r["issued"]:
            ordered.extend(q["z"])
            need(
                q["native"]["budget_bytes"] == cap * 2**20
                and q["native"]["fft_workspace_bytes"] == 0,
                "wrong physical cap",
            )
        rows.append(r)
    need(sorted(ordered) == [s["z"] for s in d["slabs"]], "incomplete cache")
    out = wd / "delivery"
    q = g.issue(cache, out)
    need(
        q["computed"] == 0
        and not q["groups"]
        and q["projection_bytes_read"] == 0
        and offline(out, d) == q["manifest_sha256"],
        "delivery was not zero-native/verified",
    )
    closed_cache(cache, d)
    return dict(
        steps=steps,
        phases=rows,
        manifest=q["manifest_sha256"],
        volume=q["volume_sha256"],
        final_workers=len(q["groups"]),
    )


def run(out):
    out = fresh(out)
    d = b.evidence()
    need(
        g.sha(M / "reference/full-corrected-512.f32") == d["volume_sha256"],
        "no retained reference",
    )
    status = oracle(d)
    runs = []
    for cap in CAPS:
        with tempfile.TemporaryDirectory(prefix="budget-frontier-", dir=M / "maintenance") as wd:
            r = real(wd, d, [(cap, 5)])
            row = r["phases"][0]
            limit = next(x for x in status["frontier"] if x["cap_mib"] == cap)
            need(
                len(row["issued"]) == limit["workers"]
                and row["actual_peak_owned_bytes"] == limit["peak_bytes"],
                "frontier mismatch",
            )
            runs.append(r)
            print(cap, len(row["issued"]), row["actual_peak_owned_bytes"], flush=True)
    with tempfile.TemporaryDirectory(prefix="budget-changing-", dir=M / "maintenance") as wd:
        mixed = real(wd, d, [(253, 1), (161, 1), (326, 1), (1, 1)])
        need(
            [r["schedule"]["reused_z"] for r in mixed["phases"]]
            == [[], [0, 283], [0, 73, 283], [0, 73, 174, 283, 401]],
            "wrong sparse resume",
        )
        need(not mixed["phases"][-1]["issued"], "unnecessary GPU launch")
    with tempfile.TemporaryDirectory(prefix="budget-reject-", dir=M / "maintenance") as tmp:
        wd = Path(tmp)
        try:
            b.warm(wd / "cache", wd / "out", 160)
        except ValueError as e:
            need("largest missing slab" in str(e), "incorrect refusal")
            why = str(e)
        else:
            raise ValueError("160 MiB accepted")
        need(not list(wd.iterdir()), "refused cold budget changed state")
    doc = dict(
        format="CFDK-GB10-budget-frontier-qualification-v1",
        parent=b.CERT,
        plan=g.PLAN,
        source_sha256={p.name: g.sha(p) for p in [Path(__file__), Path(b.__file__)]},
        independent=status,
        runs=runs,
        changed_caps=mixed,
        refused_cold_mib=160,
        refusal=why,
        scope="Frozen geometry, original 16-view arithmetic, five immutable slab units. Optimal count then peak for one-pass groups of missing slab units only. No claim over splitting slabs, changing batch or retuning. All five cold frontiers and mixed-cap resume actually ran on GB10; no elapsed-time recommendation across caps.",
    )
    with staging(out, ".budget-qualification-") as tmp:
        stage = Path(tmp) / "ready"
        stage.mkdir()
        (stage / "qualification.json").write_text(json.dumps(doc, indent=2) + "\n")
        publish(stage, out)
    print(g.sha(out / "qualification.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
