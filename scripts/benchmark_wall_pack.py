"""AB/BA on the same retained CUDA shells; both paths deliver individual flushed wall maps."""

import argparse
import json
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from axis_gauge import pinned_reader
from common import BUILD
from compare_native import sha
from edge_review import check
from remount_loss import write_doc
from wall import STATUS, measure
from wall_pack import check_stats, measure_pack, slices


def compare(parent, work, dest, parent_sha256, *, pairs=6):
    check(6 <= pairs <= 40 and pairs % 2 == 0, "supply 6..40 even AB/BA pairs")
    work.mkdir(parents=True, exist_ok=False)
    with ExitStack() as pins:

        def bind(path, expected=None):
            pins.enter_context(pinned_reader(path))
            h = sha(path)
            check(expected is None or h == expected, "input hash mismatch: " + path.name)
            return h

        key = bind(parent / "manifest.json", parent_sha256)
        src = json.loads((parent / "manifest.json").read_bytes())
        check(
            src["format"] == "CFDK-remount-witness-job-v1" and src["status"] == "compared",
            "expected compared witness measurement",
        )
        tools = {
            k: bind(BUILD / k, src["inputs_sha256"][k] if k != "wall-pack.exe" else None)
            for k in ("cone-fdk.exe", "kernels.ptx", "wall-pack.exe")
        }
        atlas = parent / "inspected/shells.f32"
        grid = parent / "inspected/scanner.grid"
        ah = bind(atlas, src["outputs_sha256"]["inspected/shells.f32"])
        gh = bind(grid, src["outputs_sha256"]["inspected/scanner.grid"])
        result = dict(
            protocol="One warm-up per path, alternating AB/BA with fresh output files. Time includes process startup, native output validation and flush. Pack also writes, hashes and flushes individual maps inside the timer. Both paths read back and hash every individual map outside the timer. Pack omits single-level sector summaries; it emits an extra packed file. No cold-cache or clock lock. Wall stage only, not CT reconstruction or end-to-end inspection.",
            gate="For each case: matching per-level bytes and status counts every run; at least 5% lower median elapsed and >=80% paired wins; no increase in peak application-owned device bytes. Publish claims only if all cases pass.",
            parent_manifest_sha256=key,
            atlas_sha256=ah,
            grid_sha256=gh,
            tools_sha256=tools,
            pairs=pairs,
            cases={},
        )
        for name, levels, budget in [
            ("two-nonnominal", [0.018, 0.022], 16),
            ("eight-level-tiled", [0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.03, 0.04], 1),
        ]:
            expected = None
            counts = None

            def trial(which, levels=levels, budget=budget):
                nonlocal expected, counts
                with TemporaryDirectory(dir=work) as temp:
                    root = Path(temp)
                    files = [root / f"level-{i:02}.wall" for i in range(len(levels))]
                    begin = time.perf_counter_ns()
                    if which == "single":
                        stats = [
                            measure(atlas, grid, f, level, budget=budget)
                            for f, level in zip(files, levels, strict=True)
                        ]
                        elapsed = (time.perf_counter_ns() - begin) / 1e6
                        check(
                            all(
                                x["redzones_verified"]
                                and x["peak_owned_bytes"] <= x["budget_bytes"] == budget << 20
                                and x["input_bytes"] == atlas.stat().st_size
                                and x["output_bytes"] == f.stat().st_size
                                for x, f in zip(stats, files, strict=True)
                            ),
                            "single accounting",
                        )
                        counts_now = [[x["status_counts"][s] for s in STATUS] for x in stats]
                        read = sum(x["input_bytes"] for x in stats)
                        owned = max(x["peak_owned_bytes"] for x in stats)
                        tiles = stats[0]["tiles"]
                        gpu = stats[0]["gpu"]
                    else:
                        pack = root / "levels.wallpack"
                        m = measure_pack(atlas, grid, pack, levels, budget=budget)
                        level_bytes = check_stats(m, grid, atlas, pack, levels, budget)
                        digests = slices(pack, level_bytes, len(levels), root)
                        elapsed = (time.perf_counter_ns() - begin) / 1e6
                        counts_now = m["status_counts_by_level"]
                        read = m["atlas_bytes_read"]
                        owned = m["peak_owned_bytes"]
                        tiles = m["tiles"]
                        gpu = m["gpu"]
                    actual = [sha(f) for f in files]
                    if which == "pack":
                        check(digests == actual, "individual output disagrees with packed bytes")
                    if expected is None:
                        expected, counts = actual, counts_now
                    check(
                        actual == expected and counts_now == counts, "byte or rejection mismatch"
                    )
                    return dict(
                        path=which,
                        elapsed_ms=elapsed,
                        atlas_bytes_read=read,
                        peak_owned_bytes=owned,
                        tiles_per_level=tiles,
                        gpu=gpu,
                    )

            warm = [trial("single"), trial("pack")]
            runs = [
                [trial(v) for v in (("single", "pack") if i % 2 == 0 else ("pack", "single"))]
                for i in range(pairs)
            ]
            order = [{x["path"]: x for x in row} for row in runs]
            med = {
                k: statistics.median(pair[k]["elapsed_ms"] for pair in order)
                for k in ("single", "pack")
            }
            wins = sum(p["pack"]["elapsed_ms"] < p["single"]["elapsed_ms"] for p in order)
            memory_ok = all(
                p["pack"]["peak_owned_bytes"] <= p["single"]["peak_owned_bytes"] for p in order
            )
            if name == "eight-level-tiled":
                check(
                    all(s["tiles_per_level"] > 1 for row in runs for s in row),
                    "tiled case did not tile",
                )
            result["cases"][name] = dict(
                levels_per_mm=levels,
                budget_mib=budget,
                warmup=warm,
                runs=runs,
                level_sha256=expected,
                status_counts_order=list(STATUS),
                status_counts=counts,
                median_ms=med,
                paired_wins=wins,
                memory_gate=memory_ok,
                passed=(
                    memory_ok and 5 * wins >= 4 * pairs and med["pack"] <= 0.95 * med["single"]
                ),
            )
        write_doc(dest, result)
        check(
            all(c["passed"] for c in result["cases"].values()),
            "performance gate missed; see recorded trials",
        )
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("parent", "work", "dest"):
        p.add_argument(n, type=Path)
    p.add_argument("--parent-sha256", required=True)
    p.add_argument("--pairs", type=int, default=6)
    doc = compare(**vars(p.parse_args()))
    print({n: c["median_ms"] for n, c in doc["cases"].items()})
