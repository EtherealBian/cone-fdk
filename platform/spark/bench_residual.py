"""Matched, position-balanced fresh-process band comparison on an already-qualified physical pair."""

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path
from statistics import median

from build_residual import HERE, ROOT, sha
from qualify_residual import BUILD, RULES, json_checked, run_gpu, volumes
from residual_witness import need
from sealed_inputs import snapshot
from spark_publication import fresh, publish, staging

BANDS = (1, 4, 7, 16)
# First-order carryover and order balanced for four treatments; repeat in reverse block order.
DESIGN = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
BUDGET = 64


def summarize(samples, bands=BANDS):
    need(len(samples) == 8 * len(bands), "incomplete benchmark")
    summary = {}
    by_block = []
    for i in range(8):
        group = [s for s in samples if s["block"] == i]
        need(
            [s["band"] for s in group] == [bands[k] for k in (*DESIGN, *reversed(DESIGN))[i]],
            "unexpected order",
        )
        by_block.append({s["band"]: s for s in group})
    for b in bands:
        group = [block[b] for block in by_block]
        owned = {s["owned_bytes"] for s in group}
        need(len(owned) == 1, "device allocation changed")
        rec = {"owned_bytes": owned.pop()}
        need(rec["owned_bytes"] <= BUDGET << 20, "exceeded budget")
        for key in ("child_elapsed_ms", "kernel_ms", "wall_ms", "read_ms"):
            v = [s[key] for s in group]
            need(all(math.isfinite(x) and x >= 0 for x in v), "invalid timing")
            center = median(v)
            rec[key] = {
                "median": center,
                "mad": median(abs(x - center) for x in v),
                "min": min(v),
                "max": max(v),
            }
        rec["paired_child_wins_against_band_1"] = (
            sum(block[b]["child_elapsed_ms"] < block[1]["child_elapsed_ms"] for block in by_block)
            if b != 1
            else None
        )
        summary[b] = rec
    best = min(bands, key=lambda b: summary[b]["child_elapsed_ms"]["median"])
    small = min(
        (
            b
            for b in bands
            if summary[b]["child_elapsed_ms"]["median"]
            <= 1.05 * summary[best]["child_elapsed_ms"]["median"]
        ),
        key=lambda b: summary[b]["owned_bytes"],
    )
    return summary, best, small


def run(build, qualification, expected, job):
    job = fresh(job)
    q = json_checked(qualification, expected)
    need(
        q["format"] == "CFDK-paired-volume-qualification-v1"
        and q["harness_sha256"] == sha(HERE / "qualify_residual.py")
        and q["build_sha256"] == BUILD
        and q["protocol_sha256"] == RULES
        and q["slices_verified_by_numpy"] == q["shape_xyz"][2],
        "not a qualified physical pair",
    )
    s = q["source"]
    case = s["case"]
    spec = volumes(case, ROOT.parent, s["parents"].get("physical_qualification"))
    paths = spec.pop("paths")
    need(spec == s, "different measured lineage")
    br = json_checked(build / "build.json", BUILD)
    shape = q["shape_xyz"]
    size = 4 * math.prod(shape)
    want = [p["gpu"] for p in q["planes"]]
    record = []
    need(len(want) == shape[2], "row coverage")
    with staging(job, ".residual-bench-") as tmp, ExitStack() as stack:
        stage = Path(tmp) / "accepted"
        stage.mkdir()
        fds = [
            stack.enter_context(snapshot(p, h, limit=512 << 20, exact=size))
            for p, h in zip(paths, s["pins"], strict=True)
        ]
        exe = stack.enter_context(
            snapshot(
                build / "volume-residual",
                br["independent_elf_sha256"][0],
                limit=4 << 20,
                executable=True,
            )
        )
        for b in BANDS:
            d = run_gpu(exe, fds, shape, b, BUDGET)
            need(d["rows"] == want, "warm-up differs from qualification")
        for i, order in enumerate((*DESIGN, *reversed(DESIGN))):
            for k in order:
                b = BANDS[k]
                d = run_gpu(exe, fds, shape, b, BUDGET)
                need(d["rows"] == want, "timed run differs from qualification")
                record.append(
                    dict(
                        block=i,
                        band=b,
                        **{
                            t: d[t]
                            for t in (
                                "owned_bytes",
                                "child_elapsed_ms",
                                "kernel_ms",
                                "read_ms",
                                "wall_ms",
                                "launches",
                            )
                        },
                    )
                )
        summary, best, small = summarize(record)
        report = dict(
            format="CFDK-residual-band-benchmark-v1",
            qualified_pair_sha256=expected,
            source_sha256=s["pins"],
            build_sha256=BUILD,
            harness_sha256=sha(__file__),
            gpu="NVIDIA GB10",
            budget_mib=BUDGET,
            bands=list(BANDS),
            warmups=len(BANDS),
            timed_samples=len(record),
            summary=summary,
            minimum_observed_child_median_band=best,
            smallest_buffer_within_five_percent_of_minimum=small,
            samples=record,
            scope="One physical pair, input bytes already SHA-verified and sealed in RAM. Fresh process, full read/copies/device computation/512-line JSON output; separately record child elapsed and kernel. No fixed clocks, no storage-cold run, no whole-CT speedup claim; selection is descriptive, no default changed.",
        )
        (stage / "benchmark.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        publish(stage, job)
        return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("qualification", type=Path)
    p.add_argument("qualification_sha256")
    p.add_argument("job", type=Path)
    a = p.parse_args()
    r = run(a.build, a.qualification, a.qualification_sha256, a.job)
    print(json.dumps(r["summary"]))
