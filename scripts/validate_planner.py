"""Compare two native scheduling policies on the published 720-view six-ROI workload."""

import argparse
import json
from math import ceil
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

from common import BUILD, DATA, REPORT, ROOT, reconstruct
from compare_native import sha
from validate_roi import require

LIST = ROOT / "fixtures/walnut-six-rois.txt"
# Freeze the independently checked pack, not the first output of the timed run.
PACK = "bf555f872da15e51ddb9501dfd503e98873bbaa414f40e2ff11992a98545ef1f"
POLICIES = ("batch-first", "min-passes")
KEYS = (
    "batch",
    "feasible_plans_scored",
    "projection_passes",
    "projection_bytes_read",
    "filter_batches",
    "launches_bp",
    "effective_budget_bytes",
    "peak_owned_bytes",
    "fft_workspace_bytes",
    "redzones_verified",
    "process_seconds",
    "wall_ms",
    "filter_ms",
    "bp_ms",
)


def run(scan, dest, scratch):
    previous = json.loads((REPORT / "multi-roi-walnut.json").read_text())
    inputs = {ext: sha(scan.with_suffix(ext)) for ext in (".f32", ".geom")}
    require(
        inputs == previous["inputs_sha256"] and sha(LIST) == previous["list_sha256"],
        "inputs or six-box manifest differ from published qualification",
    )
    size = scan.with_suffix(".f32").stat().st_size
    evidence = dict(
        dataset=previous["dataset"],
        inputs_sha256=inputs,
        list_sha256=sha(LIST),
        binaries_sha256={k: sha(BUILD / k) for k in ("cone-fdk.exe", "kernels.ptx")},
        reference_pack_sha256=PACK,
        n=512,
        fov_mm=40,
        requested_batch=16,
        timed_budget_mib=48,
        protocol="Same build and output path, one warm-up per policy, six AB/BA pairs. Subprocess time includes startup, planning, reconstruction, flush and replace. Hashes outside timer. No cache eviction or clock lock. Read bytes are logical application reads.",
        speed_gate="At least 5/6 faster pairs and at least 10% reduction in median process time at 48 MiB. Other budgets are single-run scheduling controls, not timing benchmarks.",
    )
    with TemporaryDirectory(dir=scratch) as tmp:
        out = Path(tmp) / "six.croi"
        out.touch()

        def trial(policy, cap):
            m = reconstruct(scan, out, 512, 40, cap, 16, roi_list=LIST, planner=policy)
            h = sha(out)
            require(h == PACK and m["output_bytes"] == out.stat().st_size, "wrong pack or length")
            require(
                m["views"] == 720
                and m["requested_batch"] == 16
                and m["planner"] == policy
                and m["redzones_verified"]
                and m["peak_owned_bytes"] <= m["effective_budget_bytes"] == cap << 20,
                "config, effective budget, or red zones changed",
            )
            require(
                m["projection_bytes_read"] == size * m["projection_passes"]
                and m["filter_batches"] == m["projection_passes"] * ceil(720 / m["batch"]),
                "observed reads or filter count incompatible with schedule",
            )
            env = {k: m[k] for k in ("gpu", "driver_api", "cufft_version")}
            require(evidence.setdefault("runtime", env) == env, "runtime changed")
            return {"sha256": h, **{k: m[k] for k in KEYS}}

        def pair(cap, order):
            a, b = POLICIES
            runs = {p: trial(p, cap) for p in order}
            require(
                runs[b]["projection_passes"] <= runs[a]["projection_passes"],
                "search chose more passes than the default",
            )
            if cap == 48:
                require(
                    (runs[a]["projection_passes"], runs[b]["projection_passes"]) == (3, 1),
                    "expected three passes versus one on qualification workload",
                )
            if cap == 128:
                require(
                    all(r["projection_passes"] == 1 and r["batch"] == 16 for r in runs.values()),
                    "one-pass workload should keep the requested batch",
                )
            return runs

        evidence["warmup"] = pair(48, POLICIES)
        pairs = [pair(48, POLICIES if i % 2 == 0 else POLICIES[::-1]) for i in range(6)]
        evidence["pairs"] = pairs
        a, b = POLICIES
        old, new = (median(r[p]["process_seconds"] for r in pairs) for p in POLICIES)
        wins = sum(r[b]["process_seconds"] < r[a]["process_seconds"] for r in pairs)
        evidence["time_summary"] = dict(
            batch_first_median_seconds=old,
            min_passes_median_seconds=new,
            paired_wins=wins,
            median_reduction_fraction=1 - new / old,
            speed_gate_passed=wins >= 5 and new <= 0.90 * old,
        )
        evidence["budget_controls_mib"] = {
            str(cap): pair(cap, POLICIES) for cap in (24, 32, 40, 56, 64, 128)
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf8")
    print(
        f"48 MiB, verified pack: 3 -> 1 reads; {old:.4f} -> {new:.4f} s, {wins}/6 faster. Report: {dest}"
    )
    require(
        evidence["time_summary"]["speed_gate_passed"],
        "48 MiB timing gate not met; raw observations saved",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=DATA / "full-corrected")
    p.add_argument("--out", type=Path, default=REPORT / "roi-planner-walnut.json")
    p.add_argument("--scratch", type=Path)
    a = p.parse_args()
    run(a.scan, a.out, a.scratch)
