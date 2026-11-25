"""Compare whole, fsynced FDK reconstruction on GB10 at equal geometry, PTX and budget."""

import argparse
import hashlib
import itertools
import json
import math
import os
import stat
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from build_coherent_fdk import BASE, EXTRAS, HERE, PTX, ROOT, sha
from qualify_residual import NATIVE, ASTRA, json_checked
from residual_witness import need
from sealed_inputs import snapshot
from spark_publication import fresh, staging, publish

BUILD = "6e43efe50b38893961510e8ddf3ac667cffac2fdfc39752dc7f4d97421273b9b"
PLAN = "bb25b41dae5f7fdccdc58c6692313addafca96973a44c99cc1b97ceec15bd047"
MODES = ("original", "staged", "shared")
CASES = ("dense", "slab")
MIB = 1 << 20


def schedule():
    # Each arm in each position twice; case order also alternates.
    for j, permutation in enumerate(itertools.permutations(MODES)):
        for case in CASES if j % 2 == 0 else CASES[::-1]:
            for mode in permutation:
                yield j, case, mode


def box(case, plan):
    n = plan["n"]
    need(case in CASES, "unknown case")
    z = plan["slabs"][0]
    if case == "dense":
        return [0, 0, 0, n, n, n], plan["volume_sha256"]
    return [0, 0, z["z"], n, n, z["depth"]], z["sha256"]


def validate(d, case, mode, plan):
    b, _ = box(case, plan)
    size = math.prod(b[3:]) * 4
    read = plan["projections_bytes"]
    need(mode in MODES, "mode")
    want = {
        "gpu": "NVIDIA GB10",
        "views": 720,
        "n": 512,
        "batch": 16,
        "output_shape_xyz": b[3:],
        "roi_origin_xyz": b[:3],
        "output_bytes": size,
        "projection_passes": 1,
        "projection_bytes_read": read,
        "filter_batches": 45,
        "launches_bp": 45,
        "budget_bytes": 768 * MIB,
        "redzones_verified": True,
        "slabs": 1,
    }
    if mode != "original":
        want.update(
            gb10_transport=mode,
            bulk_h2d_bytes=read if mode == "staged" else 0,
            bulk_d2h_bytes=size if mode == "staged" else 0,
        )
    need(
        all(k in d and type(d[k]) is type(v) and d[k] == v for k, v in want.items()),
        "wrong FDK geometry, transport or coverage",
    )
    need(
        0 < d["peak_owned_bytes"] <= d["budget_bytes"]
        and all(
            isinstance(d[k], (float, int)) and math.isfinite(d[k]) and d[k] >= 0
            for k in ("filter_ms", "bp_ms", "h2d_ms", "d2h_ms", "wall_ms")
        ),
        "memory or timer",
    )
    if mode != "original":
        need(
            type(d["sve_bits"]) is int
            and 128 <= d["sve_bits"] <= 2048
            and d["sve_bits"] % 128 == 0,
            "vector length",
        )
    if mode == "shared":
        need(
            d["h2d_ms"] == d["d2h_ms"] == d["host_pinned_bytes"] == 0
            and d["shared_alloc_bytes"] == size + read * 16 // 720 + 1024,
            "unexpected shared allocation or transfer",
        )
    else:
        need(
            d["host_pinned_bytes"] > 0 and d.get("shared_alloc_bytes", 0) == 0,
            "unexpected staged storage",
        )


def digest_fd(fd, offset, length):
    h = hashlib.sha256()
    while length:
        data = os.pread(fd, min(length, MIB), offset)
        need(data, "truncated reference")
        h.update(data)
        length -= len(data)
        offset += len(data)
    return h.hexdigest()


def summary(rows, plan):
    need(
        [(r["block"], r["case"], r["mode"]) for r in rows] == list(schedule()),
        "incomplete crossover",
    )
    answer = {}
    for case in CASES:
        arms = {}
        times = {}
        baseline = None
        for mode in MODES:
            group = [r for r in rows if r["case"] == case and r["mode"] == mode]
            b, h = box(case, plan)
            for r in group:
                validate(r["native"], case, mode, plan)
                need(
                    r["sha256"] == h and math.isfinite(r["elapsed_ms"]) and r["elapsed_ms"] > 0,
                    "unqualified trial",
                )
                invariant = tuple(
                    r["native"][k]
                    for k in (
                        "peak_owned_bytes",
                        "fft_workspace_bytes",
                        "filter_batches",
                        "launches_bp",
                        "batch",
                        "slab",
                        "effective_budget_bytes",
                    )
                )
                if baseline is None:
                    baseline = invariant
                need(invariant == baseline, "unequal reconstruction plan")
            ts = times[mode] = [r["elapsed_ms"] for r in group]
            m = median(ts)
            arms[mode] = dict(
                process_median_ms=m,
                mad_ms=median(abs(x - m) for x in ts),
                native_median_ms=median(r["native"]["wall_ms"] for r in group),
                peak_owned_bytes=baseline[0],
                host_pinned_bytes=group[0]["native"]["host_pinned_bytes"],
            )
        winner = "original"
        for mode in MODES[1:]:
            wins = sum(a < b for a, b in zip(times[mode], times["original"], strict=True))
            ratio = arms[mode]["process_median_ms"] / arms["original"]["process_median_ms"]
            arms[mode].update(
                wins_vs_original=wins,
                median_reduction=1 - ratio,
                clears_gate=wins >= 5 and ratio <= 0.95,
            )
            if (
                arms[mode]["clears_gate"]
                and arms[mode]["process_median_ms"] < arms[winner]["process_median_ms"]
            ):
                winner = mode
        answer[case] = dict(arms=arms, preferred=winner)
    return answer


def run(dest):
    dest = fresh(dest)
    m = ROOT.parent
    ref = m / "reference"
    fp = ROOT / "build/spark"
    plan = json_checked(ROOT / "fixtures/walnut512-slab-replay-spark-v1.json", PLAN)
    br = json_checked(m / "coherent-fdk-build-002/build.json", BUILD)
    n = json_checked(m / "walnut-gb10-001/result.json", NATIVE)
    a = json_checked(m / "walnut-astra-gb10-001/result.json", ASTRA)
    need(
        br["original"] == BASE
        and br["ptx"] == PTX
        and br["builder_sha256"] == sha(HERE / "build_coherent_fdk.py")
        and br["patch"] == {k: sha(HERE / "fdk" / k) for k in EXTRAS}
        and len(set(br["independent_elf_sha256"])) == 1
        and len(br["independent_elf_sha256"]) == 2
        and a["passed"] is True
        and n["input_sha256"] == a["input_sha256"]
        and n["output_sha256"] == [plan["volume_sha256"]] * 2,
        "upstream gates",
    )
    versions = {
        "native_qualification": NATIVE,
        "independent_astra": ASTRA,
        "protocol": PLAN,
        "build": BUILD,
        "ptx": PTX,
        "original": BASE,
    }
    warmups = []
    rows = []
    with ExitStack() as st:

        def held(p, h, limit, ex=False):
            return st.enter_context(snapshot(p, h, limit=limit, executable=ex))

        ex = held(fp / "cone-fdk", BASE, 8 * MIB, True)
        new = held(
            m / "coherent-fdk-build-002/coherent-fdk",
            br["independent_elf_sha256"][0],
            8 * MIB,
            True,
        )
        ptx = held(fp / "kernels.ptx", PTX, 4 * MIB)
        geom = held(ref / "full-corrected.geom", plan["geometry_sha256"], 4096)
        projections = held(ref / "full-corrected.f32", plan["projections_sha256"], 256 * MIB)
        volume = held(ref / "full-corrected-512.f32", plan["volume_sha256"], 512 * MIB)
        for case in CASES:
            b, h = box(case, plan)
            need(
                digest_fd(volume, b[2] * b[3] * b[4] * 4, math.prod(b[3:]) * 4) == h,
                "no retained reference for deduplication",
            )
        fdpath = lambda fd: f"/proc/self/fd/{fd}"
        # On failure retain the private work directory rather than deleting an unexplained volume.
        import tempfile

        wd = Path(tempfile.mkdtemp(prefix="fdk-crossover-", dir=m / "maintenance"))
        initial = os.lstat(wd)
        jobs = [(-1, c, m) for c in CASES for m in MODES] + list(schedule())
        for j, c, mode in jobs:
            b, h = box(c, plan)
            target = wd / f"{c}-{j}-{mode}.f32"
            elf = ex if mode == "original" else new
            cmd = [
                fdpath(elf),
                "--ptx",
                fdpath(ptx),
                "--geom",
                fdpath(geom),
                "--input",
                fdpath(projections),
                "--output",
                str(target),
                "--n",
                "512",
                "--fov",
                "40",
                "--budget",
                "768",
                "--batch",
                "16",
            ]
            if mode != "original":
                cmd += ["--gb10-transport", mode]
            if c != "dense":
                cmd += ["--roi", ",".join(map(str, b))]
            need(
                not os.path.lexists(target) and os.path.samestat(initial, wd.lstat()),
                "private workspace changed",
            )
            t = time.monotonic_ns()
            p = subprocess.run(
                cmd, capture_output=True, pass_fds=(elf, ptx, geom, projections), timeout=180
            )
            elapsed = (time.monotonic_ns() - t) / 1e6
            need(
                p.returncode == 0 and not p.stderr,
                f"{j}/{c}/{mode}: " + p.stderr.decode(errors="replace")[:600],
            )
            d = json.loads(p.stdout)
            validate(d, c, mode, plan)
            info = target.lstat()
            need(
                stat.S_ISREG(info.st_mode)
                and info.st_nlink == 1
                and info.st_size == d["output_bytes"]
                and sha(target) == h
                and os.path.samestat(info, target.lstat()),
                "FDK output mismatch; retained for diagnosis",
            )
            (rows if j >= 0 else warmups).append(
                dict(block=j, case=c, mode=mode, native=d, elapsed_ms=elapsed, sha256=h)
            )
            target.unlink()
            print(j, c, mode, round(elapsed, 2), flush=True)
        wd.rmdir()
        result = summary(rows, plan)
        doc = dict(
            format="CFDK-coherent-reconstruction-crossover-v1",
            parents=versions,
            harness_sha256=sha(__file__),
            volume_sha256=plan["volume_sha256"],
            geometry_sha256=plan["geometry_sha256"],
            projections_sha256=plan["projections_sha256"],
            warmups=warmups,
            samples=rows,
            summary=result,
            scope="Same sealed physical walnut inputs and frozen GPU PTX. Fresh child, fsynced disk output included; source snapshots and post-run SHA excluded. Warm data, unlocked clocks; two output shapes of the same specimen. No cross-device or defect accuracy claim.",
            retention="Every timed output hashed against independently retained full/slab bytes before unlinking from a private workspace. Unexplained outputs are not removed.",
        )
        with staging(dest, ".coherent-pairs-") as t:
            stage = Path(t) / "stage"
            stage.mkdir()
            (stage / "crossover.json").write_text(json.dumps(doc, indent=2) + "\n")
            publish(stage, dest)
    print(json.dumps(result))
    print(sha(dest / "crossover.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dest", type=Path)
    run(p.parse_args().dest)
