"""Paired fresh processes on one Spark: compare owned vs linked ASTRA projection input of three independent FDK witnesses."""

import argparse, fcntl, hashlib, json, math, os, resource, subprocess, sys, time
from contextlib import ExitStack
from pathlib import Path
from statistics import median
import numpy as np
import grid_oracle as g
from astra_projection_lease import projection_lease
from grouped_replay import M, sha, need, json_checked, ASTRA, SCAN
from sealed_inputs import snapshot, SEALS
from spark_publication import fresh, staging, publish

ORACLE = "4ef32216b7cba0de9b2f3f2a4c968dcdb3746cefcda71d733e878f1a9fa391fc"
SRC = Path(__file__).resolve().parent


def exercise(astra, views, pg, mode):
    sample = {}
    volumes = []

    def registered(alias, n):
        status = dict(
            line.split(":", 1)
            for line in Path("/proc/self/status").read_text().splitlines()
            if ":" in line
        )
        sample.update(
            role="same data pointer" if alias else "distinct data pointer",
            buffer_bytes=n,
            rss_anon_kib=int(status["RssAnon"].split()[0]),
        )

    start = time.monotonic_ns()
    with projection_lease(astra, pg, views, mode, registered) as pid:
        registered_ns = time.monotonic_ns()
        for n, fov, box in g.GRIDS:
            volumes.append(g.reconstruct(astra, pid, n, fov, box)[0])
    end = time.monotonic_ns()
    return volumes, sample | dict(
        total_ms=(end - start) / 1e6,
        setup_ms=(registered_ns - start) / 1e6,
        process_hwm_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def pinned():
    d = json_checked(M / "multigrid-oracle-001/qualification.json", ORACLE)
    base = json_checked(M / "walnut-astra-gb10-001/result.json", ASTRA)
    need(
        d["passed"] is True
        and d["parents"]["astra"] == ASTRA
        and d["source_sha256"] == sha(SRC / "grid_oracle.py"),
        "oracle/source parent",
    )
    return d, base


def worker(mode, gfd, sfd):
    import astra

    d, base = pinned()
    env = g.libcheck(astra, base)
    need(
        all(fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS == SEALS for fd in (gfd, sfd)),
        "sealed acquisition required",
    )
    dims, params = g.geometry(os.pread(gfd, 128, 0))
    need(os.fstat(sfd).st_size == 4 * math.prod(dims), "projection length")
    views = np.memmap("/proc/self/fd/" + str(sfd), dtype="<f4", mode="r", shape=dims)
    pg = g.proj(astra, dims, params)
    outputs, row = exercise(astra, views, pg, mode)
    keys = [hashlib.sha256(memoryview(k).cast("B")).hexdigest() for k in outputs]
    need(
        keys == [r["sha256"] for r in d["oracle_pack"]["regions"]],
        "independent ASTRA output drift",
    )
    return row | dict(mode=mode, sha256=keys, device=env["device"])


def schedule():
    return [
        (j, k)
        for j in range(-1, 6)
        for k in (("copy", "borrow") if j == -1 or j % 2 == 0 else ("borrow", "copy"))
    ]


def compare(rows):
    need([(r["block"], r["mode"]) for r in rows] == schedule(), "incomplete pairs")
    copies, loans = (
        [r for r in rows if r["block"] >= 0 and r["mode"] == k] for k in ("copy", "borrow")
    )
    for r in rows:
        need(
            r["role"] == ("distinct data pointer" if r["mode"] == "copy" else "same data pointer")
            and type(r["buffer_bytes"]) is int
            and r["buffer_bytes"] == 238694400,
            "ownership check",
        )
        need(
            all(
                type(r[k]) in (int, float) and math.isfinite(r[k]) and r[k] > 0
                for k in ("total_ms", "setup_ms", "process_hwm_kib", "rss_anon_kib", "process_ms")
            ),
            "invalid samples",
        )
    pairs = list(zip(copies, loans, strict=True))
    agg = {
        f: {k: median(r[f] for r in v) for k, v in [("copy", copies), ("borrow", loans)]}
        for f in ("total_ms", "setup_ms", "process_ms", "rss_anon_kib", "process_hwm_kib")
    }
    safe = sum(b["total_ms"] <= 1.05 * a["total_ms"] for a, b in pairs)
    mem = sum(a["rss_anon_kib"] - b["rss_anon_kib"] >= 200 * 1024 for a, b in pairs)
    proc = sum(b["process_ms"] <= 1.05 * a["process_ms"] for a, b in pairs)
    ok = (
        safe >= 5
        and proc >= 5
        and mem >= 5
        and agg["total_ms"]["borrow"] <= 1.05 * agg["total_ms"]["copy"]
        and agg["process_ms"]["borrow"] <= 1.05 * agg["process_ms"]["copy"]
    )
    return dict(
        medians=agg,
        within_5pct_pipeline=safe,
        within_5pct_process=proc,
        at_least_200MiB_transient_anon_saved=mem,
        selected=ok,
    )


def bench(out):
    out = fresh(out)
    d, base = pinned()
    rows = []
    with ExitStack() as st, staging(out, ".astra-lease-") as td:
        stage = Path(td) / "ready"
        stage.mkdir()
        fd = tuple(
            st.enter_context(snapshot(path, d["inputs"][path.name], exact=size, limit=size))
            for path, size in [
                (SCAN.with_suffix(".geom"), 49),
                (SCAN.with_suffix(".f32"), 238694400),
            ]
        )
        for j, k in schedule():
            t = time.monotonic_ns()
            proc = subprocess.run(
                [sys.executable, "-B", __file__, "worker", k, *map(str, fd)],
                pass_fds=fd,
                capture_output=True,
                timeout=90,
            )
            ms = (time.monotonic_ns() - t) / 1e6
            need(
                proc.returncode == 0 and not proc.stderr,
                "worker failed: " + proc.stderr.decode(errors="replace")[-1500:],
            )
            r = json.loads(proc.stdout)
            need(
                r["mode"] == k
                and r["device"] == base["device"]
                and r["sha256"] == [v["sha256"] for v in d["oracle_pack"]["regions"]],
                "worker receipt drift",
            )
            rows.append(r | dict(block=j, process_ms=ms))
        doc = dict(
            format="CFDK-Spark-ASTRA-projection-lease-v1",
            oracle=ORACLE,
            source={
                n: sha(SRC / n)
                for n in ("astra_projection_lease.py", "bench_astra_lease.py", "grid_oracle.py")
            },
            rows=rows,
            summary=compare(rows),
            scope="Measured walnut, three outputs. Fresh process per arm; same sealed source and loaded filesystem pages. Pipeline includes rearrangement, registration, all three ASTRA algorithms, crops and teardown. Process also includes Python imports, library provenance and all output hashes; neither includes one-time sealing or on-disk publication. RssAnon sampled at end of registration, process HWM sampled after worker reconstruction; not GPU-owned bytes.",
        )
        (stage / "benchmark.json").write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        publish(stage, out)
        return doc


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 4 and args[0] == "worker":
        print(json.dumps(worker(args[1], int(args[2]), int(args[3]))))
    else:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument("out", type=Path)
        a = p.parse_args()
        with (M / "gpu.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            print(json.dumps(bench(a.out)["summary"]))
