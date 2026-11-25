"""Replay a qualified walnut reconstruction with a GB10 workload-specific tile policy."""

import argparse
import fcntl
import json
import os
import platform
import stat
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import tune_backproject as t
from build_tile_fdk import PARENT, KERNEL, GEOM
from residual_witness import need
from sealed_inputs import snapshot
from spark_publication import fresh, publish, staging

PROFILE = "e0679bc13491589584b05e7af6a7d2b33fd7b477668c605473c0f922b5c31787"
M = t.ROOT.parent
CASES = ("dense", "slab2")
POLICIES = ("profiled", "original")
FIXTURE = t.ROOT / "fixtures/walnut512-slab-replay-spark-v1.json"


def attest(p, b, plan, n, a):
    parents = dict(build=t.BUILD, plan=t.PLAN, native=t.NATIVE, astra=t.ASTRA)
    need(
        p["format"] == "CFDK-GB10-backprojection-tile-v1"
        and p["parents"] == parents
        and p["harness_sha256"] == t.sha(t.__file__),
        "changed profile parents or harness",
    )
    need(
        b["format"] == "CFDK-GB10-tile-build-v1"
        and b["builder_sha256"] == t.sha(t.HERE / "build_tile_fdk.py")
        and (b["parent_sources"], b["kernel_parent"], b["geometry"]) == (PARENT, KERNEL, GEOM)
        and b["original"] == t.BASE
        and b["original_ptx"] == t.PTX
        and len(b["independent_products"]) == 2
        and b["independent_products"][0] == b["independent_products"][1]
        and b["tile_xyz"] == list(map(list, t.TILES)),
        "unqualified build",
    )
    need(
        plan["format"] == "CFDK-slab-replay-plan-v1"
        and (plan["n"], plan["fov_mm"], plan["batch"], plan["budget_mib"]) == (512, 40, 16, 768)
        and plan["products"] == {"cone-fdk.exe": t.BASE, "kernels.ptx": t.PTX},
        "unsupported workload",
    )
    h = n["input_sha256"]
    need(
        a["passed"] is True
        and a["astra"]["astra_version"] == "2.5.0"
        and h == a["input_sha256"]
        and p["geometry"] == plan["geometry_sha256"] == h["full-corrected.geom"]
        and p["projections"] == plan["projections_sha256"] == h["full-corrected.f32"]
        and n["output_sha256"] == [plan["volume_sha256"]] * 2
        and h["full-corrected-512.f32"] == plan["volume_sha256"],
        "reference provenance",
    )
    best, table = t.select(p["screen"], plan)
    summary = t.verdict(p["holdout"], best, plan)
    need(
        (best, table, summary) == (p["selected"], p["screen_medians"], p["summary"]),
        "profile summary mismatch",
    )
    warm = [("screen", t.tile_name(v), "screen") for v in t.TILES] + [
        (c, name, arm)
        for c in CASES
        for name, arm in [("original", "original"), (best, "selected")]
    ]
    need(
        [(r["case"], r["tile"], r["arm"]) for r in p["warmups"]] == warm
        and all(r["block"] == -1 for r in p["warmups"]),
        "warmup coverage",
    )
    for r in p["warmups"]:
        t.verify(r, plan)
    return best, summary


def route(case, policy, best, summary):
    need(case in CASES and policy in POLICIES, "unqualified case or policy")
    return (
        best
        if policy == "profiled" and summary[case]["recommendation"] == "selected"
        else "original"
    )


def evidence():
    return (
        t.json_checked(M / "tile-fdk-profile-001/profile.json", PROFILE),
        t.json_checked(M / "tile-fdk-build-001/build.json", t.BUILD),
        t.json_checked(FIXTURE, t.PLAN),
        t.json_checked(M / "walnut-gb10-001/result.json", t.NATIVE),
        t.json_checked(M / "walnut-astra-gb10-001/result.json", t.ASTRA),
    )


GUARDS = "b877d14bb4af8ff7d82072f1de20b52cee2b7cbd12a75837e33ec9472782f77b"
GUARD_ROOT = M / "tile-fdk-guards-002"


def eligible(g, best, root):
    from residual_witness import pin

    cases = ("dense", "roi", "corner", "multipass")
    names = list(map(t.tile_name, t.TILES))
    rows = g["comparisons"]
    need(
        g["format"] == "CFDK-GB10-tile-qualification-v1"
        and g["profile"] == PROFILE
        and g["build"] == t.BUILD
        and g["script_sha256"] == t.sha(t.HERE / "qualify_tiles.py")
        and g["probe_source_sha256"] == t.sha(t.HERE / "fdk/tile_attributes.cpp"),
        "unqualified diagnostic",
    )
    need(
        [(r["case"], r["tile"]) for r in rows] == [(c, name) for c in cases for name in names],
        "incomplete edge cases",
    )
    for c in cases:
        group = [r for r in rows if r["case"] == c]
        need(len({r["reference_sha256"] for r in group}) == 1, "changed case reference")
    for r in rows:
        k = r["different_bits_voxels"]
        need(
            type(k) is int
            and 0 <= k <= r["bytes"] // 4
            and (k == 0) == (pin(r["sha256"]) == pin(r["reference_sha256"]))
            and (k == 0) == (r["first_flat"] is None)
            and (k == 0) == (r["first_u32"] is None),
            "inconsistent bit witness",
        )
    ok = [
        name
        for name in names
        if all(r["different_bits_voxels"] == 0 for r in rows if r["tile"] == name)
    ]
    need(g["eligible_by_bits"] == ok and best in ok, "selected tile is unsafe")
    san = g["sanitizers"]
    need(
        [(r["case"], r["tool"], r["tile"]) for r in san]
        == [(c, tool, best) for c in ("dense", "roi") for tool in ("memcheck", "synccheck")],
        "incomplete sanitized workloads",
    )
    for r in san:
        want = next(v["reference_sha256"] for v in rows if v["case"] == r["case"])
        need(
            r["output_sha256"] == want
            and t.sha(root / (r["case"] + "-" + r["tool"] + ".log")) == r["log_sha256"],
            "wrong sanitizer result",
        )
    return ok


def run(case, out, policy, receipt_only):
    start = time.monotonic_ns()
    out = fresh(out)
    need(platform.system() == "Linux" and platform.machine() == "aarch64", "qualified Spark only")
    need(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        == "NVIDIA GB10, 12.1",
        "qualified device required",
    )
    p, b, plan, n, a = evidence()
    best, summary = attest(p, b, plan, n, a)
    eligible(t.json_checked(GUARD_ROOT / "qualification.json", GUARDS), best, GUARD_ROOT)
    tile = route(case, policy, best, summary)
    native = tile == "original"
    build = (t.ROOT / "build/spark") if native else M / "tile-fdk-build-001"
    names = ("cone-fdk", "kernels.ptx") if native else ("tile-fdk", "tile.ptx")
    pins = (t.BASE, t.PTX) if native else tuple(b["independent_products"][0][k] for k in names)
    ref = M / "reference"
    z, d, h = t.source(plan, case)
    extent = 4 * 512 * 512 * d
    offset = 4 * 512 * 512 * z
    with ExitStack() as st:

        def hold(path, pin, cap, exact=None, ex=False):
            return st.enter_context(snapshot(path, pin, limit=cap, exact=exact, executable=ex))

        exe = hold(build / names[0], pins[0], 8 * t.MIB, ex=True)
        ptx = hold(build / names[1], pins[1], 4 * t.MIB)
        geo = hold(ref / "full-corrected.geom", plan["geometry_sha256"], 4096)
        pr = hold(
            ref / "full-corrected.f32",
            plan["projections_sha256"],
            256 * t.MIB,
            plan["projections_bytes"],
        )
        if receipt_only:
            vol = hold(
                ref / "full-corrected-512.f32", plan["volume_sha256"], 512 * t.MIB, 512 * t.MIB
            )
            need(t.digest_fd(vol, offset, extent) == h, "no verified retained reference")
        with staging(out, ".spark-fdk-") as td:
            stage = Path(td) / "ready"
            stage.mkdir(mode=0o700)
            f = stage / "volume.f32"
            fd = lambda n: f"/proc/self/fd/{n}"
            cmd = [
                fd(exe),
                "--ptx",
                fd(ptx),
                "--geom",
                fd(geo),
                "--input",
                fd(pr),
                "--output",
                str(f),
                "--n",
                "512",
                "--batch",
                "16",
                "--budget",
                "768",
                "--fov",
                "40",
            ]
            if case != "dense":
                cmd += ["--roi", f"0,0,{z},512,512,{d}"]
            if not native:
                cmd += ["--gb10-tile", tile]
            begin = time.monotonic_ns()
            child = subprocess.run(
                cmd, capture_output=True, pass_fds=(exe, ptx, geo, pr), timeout=120
            )
            ms = (time.monotonic_ns() - begin) / 1e6
            need(
                child.returncode == 0 and not child.stderr,
                "native refused: " + child.stderr.decode(errors="replace")[:500],
            )
            st0 = f.lstat()
            need(
                stat.S_ISREG(st0.st_mode) and st0.st_nlink == 1 and st0.st_size == extent,
                "output type/size",
            )
            r = dict(
                case=case,
                tile=tile,
                elapsed_ms=ms,
                native=json.loads(child.stdout),
                sha256=t.sha(f),
            )
            t.verify(r, plan)
            prior = next(
                q["native"] for q in p["holdout"] if q["case"] == case and q["tile"] == tile
            )
            fixed = (
                "peak_owned_bytes",
                "host_pinned_bytes",
                "fft_workspace_bytes",
                "requested_batch",
                "driver_api",
                "cufft_version",
                "effective_budget_bytes",
                "voxel_spacing_mm",
                "planner",
                "feasible_plans_scored",
            )
            need(
                all(r["native"].get(k) == prior[k] for k in fixed),
                "different allocation, runtime or plan",
            )
            need(os.path.samestat(st0, f.lstat()), "output changed")
            retained = None
            if receipt_only:
                retained = dict(
                    file="reference/full-corrected-512.f32",
                    full_sha256=plan["volume_sha256"],
                    offset=offset,
                    length=extent,
                    range_sha256=h,
                )
                f.unlink()  # private freshly generated duplicate of the checked retained range
            doc = dict(
                format="CFDK-GB10-profiled-FDK-v1",
                source_sha256=t.sha(__file__),
                profile_sha256=PROFILE,
                eligibility_sha256=GUARDS,
                parents=p["parents"],
                products_sha256=dict(zip(names, pins)),
                input_sha256={"geometry": p["geometry"], "projections": p["projections"]},
                policy=policy,
                decision=summary[case],
                trial=r,
                receipt_only=receipt_only,
                retained_range=retained,
                verified_prepublication_ms=(time.monotonic_ns() - start) / 1e6,
                scope="Sealed single-acquisition replay. Native elapsed includes output fsync and process teardown, not authentication or output hash. Full wrapper timing ends before receipt publication and excludes the CLI lock wait.",
            )
            (stage / "result.json").write_text(json.dumps(doc, indent=2) + "\n")
            publish(stage, out)
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("out", type=Path)
    parser.add_argument("--policy", choices=POLICIES, default="profiled")
    parser.add_argument("--receipt-only", action="store_true")
    a = parser.parse_args()
    with (M / "gpu.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = run(a.case, a.out, a.policy, a.receipt_only)
        print(
            json.dumps(
                {
                    "case": a.case,
                    "tile": r["trial"]["tile"],
                    "sha256": r["trial"]["sha256"],
                    "output": str(a.out),
                }
            )
        )
