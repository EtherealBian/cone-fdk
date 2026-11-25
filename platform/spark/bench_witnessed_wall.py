"""Measure bounded AB/BA pairs on authenticated simulated profiles. Run under the Spark GPU lock."""

import argparse
import datetime
import json
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import review_wall_bench as review
import witnessed_wall as w
from sealed_inputs import snapshot


def run(store, job):
    ref = review.read(w.ROOT / "reports/wall-pack-ab.json", review.REFERENCE)
    pp = ref["parent_manifest_sha256"]
    job.mkdir(mode=0o700, exist_ok=False)
    start = datetime.datetime.now().astimezone().isoformat()
    with ExitStack() as stack:

        def snap(p, pin, limit, exe=False):
            return stack.enter_context(snapshot(p, pin, limit=limit, executable=exe))

        def obj(pin, limit):
            return snap(w.object_path(store, pin), pin, limit)

        parent = json.loads(w.fdpath(obj(pp, 4 * w.MIB)).read_bytes())
        w.need(
            parent["format"] == "CFDK-remount-witness-job-v1" and parent["status"] == "compared",
            "unaccepted parent",
        )
        witness = parent["outputs_sha256"]["body-witness.json"]
        w.need(witness == parent["inputs_sha256"]["body-witness.json"], "calibration digest")
        w.old.witness_gate(parent, json.loads(w.fdpath(obj(witness, w.MIB)).read_bytes()))
        for n in ("cone-fdk.exe", "kernels.ptx"):
            pin = parent["inputs_sha256"][n]
            w.need(ref["tools_sha256"][n] == pin, "archived origin")
            obj(pin, 16 * w.MIB)
        obj(ref["tools_sha256"]["wall-pack.exe"], 16 * w.MIB)
        ah, gh = [ref[n + "_sha256"] for n in ("atlas", "grid")]
        w.need(
            [ah, gh]
            == [
                parent["outputs_sha256"]["inspected/" + n] for n in ("shells.f32", "scanner.grid")
            ],
            "wrong scan",
        )
        atlas, grid = obj(ah, 256 * w.MIB), obj(gh, 4096)
        single, pack, ptx = [
            snap(w.BUILD / n, review.PRODUCTS[n], 16 * w.MIB, n != "kernels.ptx")
            for n in ("cone-fdk", "wall-pack", "kernels.ptx")
        ]
        cases = {}

        def trial(kind, case, seq):
            with TemporaryDirectory(dir=job) as tmp:
                d = Path(tmp)
                levels = case["levels_per_mm"]
                budget = case["budget_mib"]
                maps = [d / f"level-{j:02}.wall" for j in range(len(levels))]
                begin = time.perf_counter_ns()
                if kind == "pack":
                    out = d / "levels.wallpack"
                    meta = w.native_run(pack, ptx, atlas, grid, out, levels, budget)
                    size = w.old.check_stats(
                        meta, w.fdpath(grid), w.fdpath(atlas), out, levels, budget
                    )
                    w.old.slices(out, size, len(levels), d)
                    elapsed = time.perf_counter_ns() - begin
                    row = dict(
                        native_processes=1,
                        tiles_per_level=meta["tiles"],
                        launches=meta["kernel_launches"],
                        atlas_bytes=meta["atlas_bytes_read"],
                        wall_bytes=meta["output_bytes"],
                        gpu=meta["gpu"],
                        redzones_verified=meta["redzones_verified"],
                        peak_owned_bytes=meta["peak_owned_bytes"],
                        budget_bytes=meta["budget_bytes"],
                        status_counts=meta["status_counts_by_level"],
                    )
                else:
                    metas = []
                    for f, v in zip(maps, levels, strict=True):
                        args = [
                            w.fdpath(single),
                            "wall",
                            "--ptx",
                            w.fdpath(ptx),
                            "--input",
                            w.fdpath(atlas),
                            "--cylinder",
                            w.fdpath(grid),
                            "--output",
                            f,
                            "--threshold",
                            v,
                            "--budget",
                            budget,
                        ]
                        proc = subprocess.run(
                            list(map(str, args)),
                            pass_fds=(single, ptx, atlas, grid),
                            capture_output=True,
                            check=True,
                        )
                        metas.append(json.loads(proc.stdout))
                    elapsed = time.perf_counter_ns() - begin
                    w.need(
                        all(
                            m["gpu"] == "NVIDIA GB10"
                            and m["budget_bytes"] == budget << 20
                            and m["tiles"] == metas[0]["tiles"]
                            and m["input_bytes"] == w.fdpath(atlas).stat().st_size
                            and m["output_bytes"] == f.stat().st_size
                            for m, f in zip(metas, maps, strict=True)
                        ),
                        "legacy accounting",
                    )
                    row = dict(
                        native_processes=len(metas),
                        tiles_per_level=metas[0]["tiles"],
                        launches=sum(m["tiles"] for m in metas),
                        atlas_bytes=sum(m["input_bytes"] for m in metas),
                        wall_bytes=sum(m["output_bytes"] for m in metas),
                        gpu=metas[0]["gpu"],
                        redzones_verified=all(m["redzones_verified"] is True for m in metas),
                        peak_owned_bytes=max(m["peak_owned_bytes"] for m in metas),
                        budget_bytes=metas[0]["budget_bytes"],
                        status_counts=[
                            [m["status_counts"][v] for v in w.old.STATUS] for m in metas
                        ],
                    )
                hashes = [w.sha(p) for p in maps]
                w.need(
                    hashes == case["level_sha256"]
                    and row["status_counts"] == case["status_counts"],
                    "review outputs differ from frozen evidence",
                )
                return dict(
                    path=kind, sequence=seq, elapsed_ns=elapsed, level_sha256=hashes, **row
                )

        for name in review.EXPECTED:
            c = ref["cases"][name]
            rows = []
            for j in range(7):
                order = ["pack", "single"] if j > 0 and j % 2 == 0 else ["single", "pack"]
                rows.append([trial(kind, c, 2 * j + i) for i, kind in enumerate(order)])
            cases[name] = dict(
                warmup=rows[0],
                pairs=rows[1:],
                levels_per_mm=c["levels_per_mm"],
                budget_mib=c["budget_mib"],
            )
    doc = dict(
        format="CFDK-GB10-wall-stage-pairs-v1",
        started_at=start,
        ended_at=datetime.datetime.now().astimezone().isoformat(),
        parent_manifest_sha256=pp,
        atlas_sha256=ah,
        grid_sha256=gh,
        product_sha256=review.PRODUCTS,
        reference_sha256=review.REFERENCE,
        measurement_code_sha256=w.sha(Path(__file__)),
        adapter_sha256=w.sha(w.HERE / "witnessed_wall.py"),
        cases=cases,
        scope="For both: sealed inputs prepared before timing; process startup, native writes/fsync included. Packed path also verifies layout, splits, hashes and fsyncs individual files inside timer. Final per-file readback and rejection tallies are checked outside timer. Single-level native records include additional sector statistics. No cold-cache or GPU clock lock.",
    )
    w.old.write_doc(job / "pairs.json", doc)
    summary = review.assess(doc, ref)
    w.old.write_doc(job / "summary.json", summary)
    print(json.dumps(summary))
    return summary["all_gates_passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("store", type=Path)
    p.add_argument("job", type=Path)
    raise SystemExit(0 if run(**vars(p.parse_args())) else 2)
