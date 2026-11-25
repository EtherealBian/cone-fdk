"""One atlas upload per tile, reused by all edge levels on the original radial_wall CUDA kernel."""

import argparse
import hashlib
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path

from axis_gauge import pinned_reader
from common import BUILD
from compare_native import sha
from edge_review import check, sampling
from remount_loss import write_doc
from wall import STATUS, run_native_json


def measure_pack(atlas, grid, output, levels, *, budget=16, build=BUILD):
    cmd = [
        build / "wall-pack.exe",
        "--ptx",
        build / "kernels.ptx",
        "--input",
        atlas,
        "--cylinder",
        grid,
        "--output",
        output,
        "--budget",
        budget,
    ]
    for v in levels:
        cmd.extend(["--level", v])
    return run_native_json(cmd)


def check_stats(meta, grid, atlas, pack, levels, budget):
    w, h, d, *_ = sampling(grid)
    k = len(levels)
    check(
        meta["format"] == "CFDK-wall-pack-v1"
        and 1 <= k <= 16
        and list(levels) == sorted(set(levels))
        and all(math.isfinite(v) and v > 0 for v in levels)
        and meta["levels_per_mm"] == list(levels)
        and meta["shape_ha"] == [h, w]
        and meta["shells"] == d,
        "native wall-pack layout",
    )
    counts = meta["status_counts_by_level"]
    check(
        meta["redzones_verified"] is True
        and 0 < meta["peak_owned_bytes"] <= meta["budget_bytes"] == budget << 20
        and 1 <= meta["tiles"] <= w * h
        and meta["kernel_launches"] == k * meta["tiles"]
        and len(counts) == k
        and all(
            len(c) == len(STATUS)
            and all(type(v) is int and v >= 0 for v in c)
            and sum(c) == h * w
            for c in counts
        )
        and meta["atlas_bytes_read"]
        == meta["atlas_payload_h2d_bytes"]
        == atlas.stat().st_size
        == 4 * w * h * d
        and meta["wall_payload_d2h_bytes"]
        == meta["output_bytes"]
        == pack.stat().st_size
        == 32 * w * h * k,
        "native wall-pack accounting",
    )
    return h * w * 32


def slices(pack, level_bytes, count, folder=None):
    """Digest each raw wall map in level order; optionally materialize and flush fresh files."""
    check(
        level_bytes > 0 and 1 <= count <= 16 and pack.stat().st_size == level_bytes * count,
        "wall pack length mismatch",
    )
    hashes = []
    with pack.open("rb") as f:
        for i in range(count):
            with ExitStack() as stack:
                dst = (
                    stack.enter_context((folder / f"level-{i:02}.wall").open("xb"))
                    if folder is not None
                    else None
                )
                d = hashlib.sha256()
                pending = level_bytes
                while pending:
                    b = f.read(min(pending, 1 << 20))
                    check(b, "truncated wall pack")
                    pending -= len(b)
                    d.update(b)
                    if dst is not None:
                        dst.write(b)
                if dst is not None:
                    dst.flush()
                    os.fsync(dst.fileno())
                hashes.append(d.hexdigest())
        check(not f.read(1), "wall pack has trailing data")
    return hashes


def run(atlas, grid, output, levels, *, budget=16, unpack=False):
    atlas, grid, out = map(lambda p: Path(p).resolve(), (atlas, grid, output))
    paths = {
        "atlas": atlas,
        "grid": grid,
        **{n: BUILD / n for n in ("wall-pack.exe", "kernels.ptx")},
    }
    check(not any(p.is_relative_to(out) for p in paths.values()), "output contains an input")
    with ExitStack() as held:
        for p in paths.values():
            held.enter_context(pinned_reader(p))
        pins = {k: sha(p) for k, p in paths.items()}
        out.mkdir(parents=True, exist_ok=False)
        pack = out / "levels.wallpack"
        meta = measure_pack(atlas, grid, pack, levels, budget=budget)
        held.enter_context(pinned_reader(pack))
        n = check_stats(meta, grid, atlas, pack, levels, budget)
        digests = slices(pack, n, len(levels), out if unpack else None)
        doc = dict(
            format="CFDK-wall-pack-job-v1",
            inputs_sha256=pins,
            pack_sha256=sha(pack),
            native=meta,
            level_bytes=n,
            level_sha256=digests,
            materialized=unpack,
        )
        temp = out / "manifest.partial.json"
        write_doc(temp, doc)
        temp.replace(out / "manifest.json")
        return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("atlas", "grid", "output"):
        p.add_argument(n, type=Path)
    p.add_argument("--levels", type=float, nargs="+", required=True)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--unpack", action="store_true")
    try:
        doc = run(**vars(p.parse_args()))
    except (ValueError, RuntimeError, OSError, KeyError) as err:
        p.exit(1, f"wall pack failed: {err}\n")
    print(json.dumps(doc["native"]))
