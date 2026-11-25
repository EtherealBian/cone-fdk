"""Bounded GB10 transport for the frozen paired-volume residual protocol.

Each fresh process uses either device+host staging or a single mapped host allocation per
buffer. Work buffers are counted once. This is not driver/process RSS or peak system RAM.
"""

import argparse
import hashlib
import json
import math
import struct
import subprocess
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from statistics import median
from build_residual import HERE, ROOT, RULES, sha
from build_mapped_residual import FROZEN
from qualify_residual import BUILD, json_checked, run_gpu, volumes
from residual_witness import need
from sealed_inputs import snapshot
from spark_publication import fresh, publish, staging

MAPPED = "e5be348f8a5873d2ebc72246ab598b6353dfd618263b53aeb8876afcae263216"
WALNUT = "12f2f81c0d88d75b05fd824876b94cc236d63b5613260341c2917998a5db3349"
MIB = 1 << 20
BUDGET = 96
MODES = ("copy", "mapped")
BANDS = (4, 16)


def signature(rows):
    # All integer counts, signed zeros, and IEEE-754 binary64 values matter.
    dig = hashlib.sha256()
    for row in rows:
        need(
            len(row) == 12
            and all(type(row[i]) is int and 0 <= row[i] < 2**32 for i in (0, 1, 2, 3, 4, 5, 7))
            and all(
                type(row[i]) in (float, int) and math.isfinite(row[i]) for i in (6, 8, 9, 10, 11)
            ),
            "residual plane ABI",
        )
        dig.update(struct.pack("<6IdI4d", *row))
    return dig.hexdigest()


def memory(shape, band, mode):
    need(
        len(shape) == 3
        and all(type(x) is int and 1 <= x <= 2048 for x in shape)
        and math.prod(shape) <= 134217728
        and type(band) is int
        and 1 <= band <= 64
        and mode in MODES,
        "unsupported grid/transport",
    )
    nx, ny, nz = shape
    plane = nx * ny
    tiles = (plane + 1023) // 1024
    band = min(band, nz)
    payload = band * (plane * 8 + tiles * 72)
    return payload * (1 if mode == "mapped" else 2) + 1536 + nz * 72


def mapped_account(r, s, band, budget):
    total = memory(s, band, "mapped")
    n = math.prod(s)
    nz = s[2]
    want = dict(
        format="CFDK-GB10-mapped-residual-v1",
        gpu="NVIDIA GB10",
        shape_xyz=s,
        band=min(nz, band),
        tile_size=1024,
        partial_bytes=72,
        mapped_bytes=total - nz * 72,
        owned_total_bytes=total,
        budget_bytes=budget * MIB,
        read_bytes=8 * n,
        h2d_bytes=0,
        d2h_bytes=0,
        launches=(nz + band - 1) // band,
        guard_checks=6,
        can_map_host=1,
        uva=1,
    )
    need(
        total <= budget * MIB
        and all(type(r[k]) is type(v) and r[k] == v for k, v in want.items()),
        "mapped bounds or transport accounting",
    )
    need(
        all(
            type(r[k]) is int and r[k] in (0, 1)
            for k in ("pageable_access", "uses_host_page_tables", "host_register_supported")
        )
        and len(r["rows"]) == nz
        and all(
            type(r[k]) in (float, int) and math.isfinite(r[k]) and r[k] >= 0
            for k in ("read_ms", "kernel_ms", "wall_ms")
        ),
        "invalid mapped runtime output",
    )


def run(mode, exe, fds, shape, band, budget):
    owned = memory(shape, band, mode)
    need(
        type(budget) is int and 1 <= budget <= 1024 and owned <= budget * MIB,
        "combined application work buffers exceed budget",
    )
    if mode == "copy":
        r = run_gpu(exe, fds, shape, band, budget)
        need(r["owned_bytes"] + r["host_payload_bytes"] == owned, "copy allocation accounting")
    else:
        files = (exe, *fds)
        cmd = [*(f"/proc/self/fd/{fd}" for fd in files), *map(str, (*shape, band, budget))]
        t = time.monotonic_ns()
        p = subprocess.run(cmd, pass_fds=files, capture_output=True, timeout=180)
        elapsed = (time.monotonic_ns() - t) / 1e6
        need(
            p.returncode == 0 and not p.stderr,
            "mapped runtime refusal: " + p.stderr.decode(errors="replace")[:300],
        )
        r = json.loads(p.stdout)
        mapped_account(r, shape, band, budget)
        r["child_elapsed_ms"] = elapsed
    r["combined_work_bytes"] = owned
    return r


@contextmanager
def held(case, qualification, expected):
    m = ROOT.parent
    q = json_checked(qualification, expected)
    need(
        q["format"] == "CFDK-paired-volume-qualification-v1"
        and q["build_sha256"] == BUILD
        and q["protocol_sha256"] == RULES
        and q["harness_sha256"] == sha(HERE / "qualify_residual.py"),
        "unqualified volume pair",
    )
    spec = volumes(case, m, q["source"]["parents"].get("physical_qualification"))
    paths = spec.pop("paths")
    need(
        spec == q["source"]
        and q["shape_xyz"] == spec["shape"]
        and q["slices_verified_by_numpy"] == spec["shape"][2] == len(q["planes"]),
        "physical source/coverage",
    )
    size = math.prod(spec["shape"]) * 4
    builds = {
        "mapped": (m / "mapped-residual-build-001", "mapped-residual", MAPPED),
        "copy": (m / "residual-builds-001/qualified", "volume-residual", BUILD),
    }
    exes = {}
    with ExitStack() as stack:
        for mode, (build, name, h) in builds.items():
            report = json_checked(build / "build.json", h)
            pins = report["independent_elf_sha256"]
            need(
                len(pins) == 2 and pins[0] == pins[1] and report["protocol_sha256"] == RULES,
                "build gate",
            )
            if mode == "mapped":
                need(
                    report["kernel_source_sha256"] == FROZEN
                    and report["transport_sha256"] == sha(HERE / "residuals/mapped_residual.cu"),
                    "mapped source",
                )
            exes[mode] = stack.enter_context(
                snapshot(build / name, pins[0], limit=4 * MIB, executable=True)
            )
        fds = [
            stack.enter_context(snapshot(p, h, limit=512 * MIB, exact=size))
            for p, h in zip(paths, spec["pins"], strict=True)
        ]
        yield spec, exes, fds, signature([p["gpu"] for p in q["planes"]])


def design():
    for block in range(6):
        for band in BANDS if block % 2 == 0 else BANDS[::-1]:
            for mode in MODES if (block + BANDS.index(band)) % 2 == 0 else MODES[::-1]:
                yield block, band, mode


def decision(rows, shape, expected):
    need(
        [(r["block"], r["band"], r["mode"]) for r in rows] == list(design()),
        "incomplete or unbalanced paired design",
    )
    summary = {}
    for band in BANDS:
        med = {}
        v = {}
        by_mode = {}
        for mode in MODES:
            group = [r for r in rows if (r["band"], r["mode"]) == (band, mode)]
            owned = memory(shape, band, mode)
            expected_h2d = 8 * math.prod(shape) if mode == "copy" else 0
            expected_d2h = (
                72 * shape[2] * ((shape[0] * shape[1] + 1023) // 1024) if mode == "copy" else 0
            )
            need(
                all(
                    r["rows_sha256"] == expected
                    and r["combined_work_bytes"] == owned
                    and r["h2d_bytes"] == expected_h2d
                    and r["d2h_bytes"] == expected_d2h
                    and 0 < r["child_elapsed_ms"]
                    and math.isfinite(r["child_elapsed_ms"])
                    for r in group
                ),
                "timed samples or row witnesses",
            )
            v[mode] = [r["child_elapsed_ms"] for r in group]
            med[mode] = median(v[mode])
            by_mode[mode] = dict(
                median_ms=med[mode],
                mad_ms=median(abs(x - med[mode]) for x in v[mode]),
                combined_work_bytes=owned,
                h2d_bytes=expected_h2d,
                d2h_bytes=expected_d2h,
            )
        wins = sum(a < b for a, b in zip(v["mapped"], v["copy"], strict=True))
        preferred = "mapped" if wins >= 5 and med["mapped"] <= 0.95 * med["copy"] else "copy"
        summary[str(band)] = dict(
            methods=by_mode,
            mapped_wins=wins,
            speed_ratio=med["copy"] / med["mapped"],
            latency_choice=preferred,
        )
    return summary


def benchmark(dest):
    dest = fresh(dest)
    qual = ROOT.parent / "residual-walnut-001/qualification.json"
    rows = []
    attrs = None
    with held("walnut720", qual, WALNUT) as (spec, exes, fds, expected):
        shape = spec["shape"]
        runs = [(-1, b, m) for b in BANDS for m in MODES] + list(design())
        for block, b, mode in runs:
            r = run(mode, exes[mode], fds, shape, b, BUDGET)
            sig = signature(r["rows"])
            need(sig == expected, "measured physical rows differ")
            if mode == "mapped":
                attrs = {
                    k: r[k]
                    for k in [
                        "can_map_host",
                        "uva",
                        "pageable_access",
                        "uses_host_page_tables",
                        "host_register_supported",
                    ]
                }
            if block >= 0:
                rows.append(
                    dict(
                        block=block,
                        band=b,
                        mode=mode,
                        rows_sha256=sig,
                        **{
                            k: r[k]
                            for k in (
                                "combined_work_bytes",
                                "h2d_bytes",
                                "d2h_bytes",
                                "child_elapsed_ms",
                                "kernel_ms",
                                "read_ms",
                                "wall_ms",
                                "launches",
                                "guard_checks",
                            )
                        },
                    )
                )
    summary = decision(rows, shape, expected)
    doc = dict(
        format="CFDK-GB10-residual-transport-pairs-v1",
        case="walnut720",
        physical_qualification=WALNUT,
        volume_sha256=spec["pins"],
        shape=shape,
        rows_sha256=expected,
        mapped_build=MAPPED,
        copy_build=BUILD,
        application_budget_mib=BUDGET,
        warmups=4,
        device_attributes=attrs,
        harness_sha256=sha(__file__),
        samples=rows,
        summary=summary,
        scope="Warm sealed inputs, same CUDA residual kernel. Fresh process elapsed includes input reads, launch, reduction, synchronization, JSON output and exit; excludes initial SHA snapshots. Buffer accounting excludes source memfds, OS caching, context and allocator overhead. No reconstruction or accuracy speed claim.",
    )
    with staging(dest, ".transport-") as t:
        stage = Path(t) / "stage"
        stage.mkdir()
        (stage / "pairs.json").write_text(json.dumps(doc, indent=2) + "\n")
        publish(stage, dest)
    print(json.dumps(summary))
    print(sha(dest / "pairs.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dest", type=Path)
    benchmark(p.parse_args().dest)
