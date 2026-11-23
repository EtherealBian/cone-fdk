"""Interleave two native releases on the same volume, including validation + durable publication."""

import argparse
import datetime
import hashlib
import json
import pathlib
import statistics
import tempfile

from common import BUILD, DATA, REPORT, reconstruct


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def compare(old, new, pairs, pilot, dest):
    if pairs < 2 or pairs % 2 or (not pilot and pairs < 6):
        raise ValueError("use an even pair count (at least six for release)")
    dirs = {"old": old.resolve(), "new": new.resolve()}
    result = dict(
        date=datetime.datetime.now().astimezone().isoformat(),
        pilot=pilot,
        pairs=pairs,
        n=512,
        batch=16,
        budget_mib=768,
        binaries={
            k: {f: sha(d / f) for f in ("cone-fdk.exe", "kernels.ptx")} for k, d in dirs.items()
        },
        protocol="Separate processes; same output path, start with existing output, native flush + atomic replace. One warm-up per build and input; AB/BA. Full file SHA-256 outside timer. No cache eviction or clock lock.",
        gate="For each input: >=5% lower median native wall and process time, >=ceil(0.8*N) paired wins in both; same output bytes and no increase in engine-owned device bytes.",
        cases={},
    )
    with tempfile.TemporaryDirectory(dir=DATA) as folder:
        out = pathlib.Path(folder) / "volume.f32"
        out.touch()
        for name, fov in [("phantom", 32), ("full-corrected", 40)]:
            scan = DATA / name
            case = dict(
                fov_mm=fov,
                projections_sha256=sha(scan.with_suffix(".f32")),
                geom_sha256=sha(scan.with_suffix(".geom")),
                warmup={},
                runs=[],
            )
            result["cases"][name] = case
            expected = None

            def run(label, scan=scan, fov=fov, name=name):
                nonlocal expected
                m = reconstruct(scan, out, 512, fov, 768, 16, build=dirs[label])
                h = sha(out)
                if expected is None:
                    expected = h
                if (
                    h != expected
                    or out.stat().st_size != 4 * 512**3
                    or not m["redzones_verified"]
                    or m["batch"] != 16
                    or m["slabs"] != 1
                ):
                    raise RuntimeError(f"{name}: correctness / memory-plan gate failed")
                return dict(build=label, output_sha256=h, **m)

            for label in dirs:
                case["warmup"][label] = run(label)
            for i in range(pairs):
                case["runs"].append(
                    [run(k) for k in (("new", "old") if i % 2 else ("old", "new"))]
                )
            stats = {}
            for field in (
                "wall_ms",
                "process_seconds",
                "bp_ms",
                "filter_ms",
                "h2d_ms",
                "d2h_ms",
                "peak_owned_bytes",
            ):
                values = [{m["build"]: m[field] for m in pair} for pair in case["runs"]]
                a, b = (statistics.median(v[k] for v in values) for k in dirs)
                stats[field] = dict(
                    old=a,
                    new=b,
                    reduction_pct=100 * (1 - b / a),
                    new_wins=sum(v["new"] < v["old"] for v in values),
                )
            case["medians"] = stats
            case["passed"] = all(
                stats[f]["reduction_pct"] >= 5 and stats[f]["new_wins"] >= (4 * pairs + 4) // 5
                for f in ("wall_ms", "process_seconds")
            ) and all(
                next(m for m in pair if m["build"] == "new")["peak_owned_bytes"]
                <= next(m for m in pair if m["build"] == "old")["peak_owned_bytes"]
                for pair in case["runs"]
            )
            print(name, json.dumps(stats), flush=True)
    result["passed"] = all(c["passed"] for c in result["cases"].values())
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    if not pilot and not result["passed"]:
        raise SystemExit("release comparison gate failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=pathlib.Path, help="directory with old EXE and PTX")
    p.add_argument("--candidate", type=pathlib.Path, default=BUILD)
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--pilot", action="store_true", help="exploratory; cannot qualify a release")
    p.add_argument("--output", type=pathlib.Path, help="JSON evidence path")
    a = p.parse_args()
    compare(
        a.baseline,
        a.candidate,
        a.pairs,
        a.pilot,
        a.output or (REPORT / ("pilot.json" if a.pilot else "native-ab.json")),
    )
