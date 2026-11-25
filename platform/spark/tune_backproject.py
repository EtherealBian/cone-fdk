"""Freeze a GB10 backprojection tile using one slab; test it against original FDK on separate outputs."""

import argparse
import itertools
import json
import math
import os
import stat
import subprocess
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from bench_coherent_fdk import ASTRA, NATIVE, PLAN, digest_fd
from build_tile_fdk import BASE, HERE, PTX, ROOT, TILES, sha
from qualify_residual import json_checked
from residual_witness import need
from sealed_inputs import snapshot
from spark_publication import fresh, staging, publish

BUILD = "0db809f6d92ace06c21c0ae8d78195710b59f246420b7612159555b2a2203d5e"
MIB = 1 << 20


def tile_name(t):
    return "x".join(map(str, t))


def schedule():
    # Each tile appears once per block; rotate by a third of the table, reverse alternating blocks.
    for j in range(3):
        order = list(TILES[j * 3 :] + TILES[: j * 3])
        for t in order if j != 1 else order[::-1]:
            yield j, tile_name(t)


def heldout():
    for j in range(6):
        for k, c in enumerate(("dense", "slab2") if j % 2 == 0 else ("slab2", "dense")):
            for m in (
                ("original", "selected")
                if (j + int(c == "slab2")) % 2 == 0
                else ("selected", "original")
            ):
                yield j, c, m


def source(plan, c):
    n = plan["n"]
    need(c in ("screen", "dense", "slab2"), "unknown output")
    if c == "dense":
        return 0, n, plan["volume_sha256"]
    s = plan["slabs"][1 if c == "screen" else 2]
    return s["z"], s["depth"], s["sha256"]


def verify(r, plan):
    c = r["case"]
    tile = r["tile"]
    z, depth, h = source(plan, c)
    d = r["native"]
    n = plan["n"]
    want = dict(
        gpu="NVIDIA GB10",
        output_shape_xyz=[n, n, depth],
        roi_origin_xyz=[0, 0, z],
        views=720,
        n=n,
        batch=16,
        output_bytes=n * n * depth * 4,
        slabs=1,
        slab=depth,
        projection_passes=1,
        projection_bytes_read=plan["projections_bytes"],
        filter_batches=45,
        launches_bp=45,
        budget_bytes=768 * MIB,
        redzones_verified=True,
    )
    if tile != "original":
        need(tile in {tile_name(x) for x in TILES}, "unknown tile")
        want["gb10_tile"] = [int(x) for x in tile.split("x")]
    else:
        need("gb10_tile" not in d, "wrong baseline")
    need(
        r["sha256"] == h and all(type(d[k]) is type(v) and d[k] == v for k, v in want.items()),
        "different output, plan or tile",
    )
    need(
        all(
            type(v) in (int, float) and math.isfinite(v) and v > 0
            for v in [r["elapsed_ms"], d["bp_ms"], d["wall_ms"]]
        )
        and d["peak_owned_bytes"] <= 768 * MIB,
        "bad timers or memory",
    )


def select(rows, plan):
    need(
        [(r["block"], r["tile"]) for r in rows] == list(schedule())
        and all(r["case"] == "screen" for r in rows),
        "incomplete tuning design",
    )
    [verify(r, plan) for r in rows]
    table = {}
    for tile in map(tile_name, TILES):
        group = [r for r in rows if r["tile"] == tile]
        table[tile] = dict(
            bp_ms=median(r["native"]["bp_ms"] for r in group),
            process_ms=median(r["elapsed_ms"] for r in group),
        )
    # Select only on screen kernel time, never choose retrospectively on a holdout.
    return min(table, key=lambda t: (table[t]["bp_ms"], t)), table


def verdict(rows, best, plan):
    need(
        [(r["block"], r["case"], r["arm"]) for r in rows] == list(heldout()), "incomplete holdout"
    )
    out = {}
    for r in rows:
        verify(r, plan)
        need(r["tile"] == ("original" if r["arm"] == "original" else best), "selection changed")
    for c in ("dense", "slab2"):
        base, new = (
            [r for r in rows if r["case"] == c and r["arm"] == arm]
            for arm in ("original", "selected")
        )
        dest = {}
        need(
            len(
                {
                    (
                        r["native"]["peak_owned_bytes"],
                        r["native"]["host_pinned_bytes"],
                        r["native"]["fft_workspace_bytes"],
                        r["native"]["launches_bp"],
                    )
                    for r in base + new
                }
            )
            == 1,
            "unequal buffers or plan",
        )
        for key in ("process", "backprojection"):
            f = (
                (lambda r: r["elapsed_ms"])
                if key == "process"
                else (lambda r: r["native"]["bp_ms"])
            )
            a, b = [list(map(f, group)) for group in (base, new)]
            x, y = median(a), median(b)
            wins = sum(v < u for u, v in zip(a, b, strict=True))
            dest[key] = dict(
                original_median_ms=x,
                selected_median_ms=y,
                original_mad_ms=median(abs(v - x) for v in a),
                selected_mad_ms=median(abs(v - y) for v in b),
                paired_wins=wins,
                median_reduction=1 - y / x,
                gate=wins >= 5 and y <= 0.95 * x,
            )
        dest["recommendation"] = (
            "selected" if all(v["gate"] for v in dest.values()) else "original"
        )
        out[c] = dest
    return out


def run(dest):
    dest = fresh(dest)
    m = ROOT.parent
    ref = m / "reference"
    fp = ROOT / "build/spark"
    bp = m / "tile-fdk-build-001"
    plan = json_checked(ROOT / "fixtures/walnut512-slab-replay-spark-v1.json", PLAN)
    br = json_checked(bp / "build.json", BUILD)
    n = json_checked(m / "walnut-gb10-001/result.json", NATIVE)
    a = json_checked(m / "walnut-astra-gb10-001/result.json", ASTRA)
    need(
        a["passed"] is True
        and n["input_sha256"] == a["input_sha256"]
        and plan["volume_sha256"] == n["output_sha256"][0] == n["output_sha256"][1]
        and br["builder_sha256"] == sha(HERE / "build_tile_fdk.py")
        and br["independent_products"][0] == br["independent_products"][1]
        and br["tile_xyz"] == list(map(list, TILES)),
        "changed source or build",
    )
    products = br["independent_products"][0]
    with ExitStack() as st:

        def hold(p, h, cap, ex=False):
            return st.enter_context(snapshot(p, h, limit=cap, executable=ex))

        native = hold(fp / "cone-fdk", BASE, 8 * MIB, True)
        oldptx = hold(fp / "kernels.ptx", PTX, 4 * MIB)
        elf = hold(bp / "tile-fdk", products["tile-fdk"], 8 * MIB, True)
        ptx = hold(bp / "tile.ptx", products["tile.ptx"], 4 * MIB)
        geo = hold(ref / "full-corrected.geom", plan["geometry_sha256"], 4096)
        projections = hold(ref / "full-corrected.f32", plan["projections_sha256"], 256 * MIB)
        vol = hold(ref / "full-corrected-512.f32", plan["volume_sha256"], 512 * MIB)
        for c in ("screen", "dense", "slab2"):
            z, depth, h = source(plan, c)
            need(
                digest_fd(vol, 4 * 512**2 * z, 4 * 512**2 * depth) == h,
                "no authenticated retained reference",
            )
        wd = Path(tempfile.mkdtemp(prefix="fdk-tile-", dir=m / "maintenance"))
        initial = wd.lstat()
        fdpath = lambda fd: f"/proc/self/fd/{fd}"

        def trial(j, c, tile, arm):
            z, depth, h = source(plan, c)
            x, y = (native, oldptx) if tile == "original" else (elf, ptx)
            f = wd / "output.f32"
            need(
                not os.path.lexists(f) and os.path.samestat(wd.lstat(), initial),
                "private workspace changed",
            )
            cmd = [
                fdpath(x),
                "--ptx",
                fdpath(y),
                "--geom",
                fdpath(geo),
                "--input",
                fdpath(projections),
                "--output",
                str(f),
                "--n",
                "512",
                "--fov",
                "40",
                "--batch",
                "16",
                "--budget",
                "768",
            ]
            if c != "dense":
                cmd += ["--roi", f"0,0,{z},512,512,{depth}"]
            if tile != "original":
                cmd += ["--gb10-tile", tile]
            start = time.monotonic_ns()
            p = subprocess.run(
                cmd, pass_fds=(x, y, geo, projections), capture_output=True, timeout=120
            )
            elapsed = (time.monotonic_ns() - start) / 1e6
            need(p.returncode == 0 and not p.stderr, p.stderr.decode(errors="replace")[:500])
            info = f.lstat()
            need(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and info.st_size == 4 * 512**2 * depth,
                "output type/length",
            )
            r = dict(
                block=j,
                case=c,
                tile=tile,
                arm=arm,
                elapsed_ms=elapsed,
                native=json.loads(p.stdout),
                sha256=sha(f),
            )
            verify(r, plan)
            need(os.path.samestat(info, f.lstat()), "output changed")
            f.unlink()
            print(j, c, tile, round(r["native"]["bp_ms"], 2), round(elapsed, 2), flush=True)
            return r

        warm = [trial(-1, "screen", tile_name(t), "screen") for t in TILES]
        screen = [trial(j, "screen", t, "screen") for j, t in schedule()]
        best, table = select(screen, plan)
        print("frozen selection", best, flush=True)
        warm += [
            trial(-1, c, t, arm)
            for c in ("dense", "slab2")
            for arm, t in [("original", "original"), ("selected", best)]
        ]
        samples = [
            trial(j, c, ("original" if arm == "original" else best), arm)
            for j, c, arm in heldout()
        ]
        results = verdict(samples, best, plan)
        wd.rmdir()
        doc = dict(
            format="CFDK-GB10-backprojection-tile-v1",
            parents=dict(build=BUILD, plan=PLAN, native=NATIVE, astra=ASTRA),
            harness_sha256=sha(__file__),
            geometry=plan["geometry_sha256"],
            projections=plan["projections_sha256"],
            warmups=warm,
            screen=screen,
            screen_medians=table,
            selected=best,
            holdout=samples,
            summary=results,
            scope="Screen uses the second slab; validate on full volume and third slab, six balanced fresh-process pairs per output. These are separate reconstruction requests of one physical specimen; the dense volume includes the tuning slab. Fixed batch, output fsync included; reference hashing excluded. No claim of independent specimens, defect accuracy or general speedup.",
        )
        with staging(dest, ".tile-run-") as td:
            stage = Path(td) / "qualified"
            stage.mkdir()
            (stage / "profile.json").write_text(json.dumps(doc, indent=2) + "\n")
            publish(stage, dest)
    print(json.dumps(results))
    print(sha(dest / "profile.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
