"""Recalculate a frozen witness parent's radial walls and edge envelope on GB10.

The archived producer and PTX are authenticated DATA, never executed. A separately
pinned Linux binary and PTX consume sealed snapshots. No Windows share-mode emulation.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

from sealed_inputs import snapshot
from spark_publication import fresh, publish, staging

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = ROOT / "scripts"
BUILD = ROOT / "build/spark"
SOURCE = {
    "edge_review_packed.py": "72ff84e39bb04de9f894910594378894ba9231288d8d4649925e663fc61c890d",
    "edge_review.py": "125fdcc7abc7adcdb520f62bf7660f38975300d76dfcaf44c369aff334f24c7a",
    "wall_pack.py": "5ec5a9aecbf1c0e4b7ffe126b75d4b7cabf879d8b384ceb3c16e0be3c3b3ada6",
    "wall.py": "b0c1d8f832c21c8d886df961d455b6e476c98aa8514c30547b73ef872f7c2497",
}
NATIVE = {
    "wall-pack": "6351b4db3aa52e1df6a9651ccc329dcb238c45c6cdbc498dbd0c6cef9a72e7dc",
    "kernels.ptx": "2ff008d243a421c97c93f6c94031484d9ec165084855173386e6a491dce2d167",
}
MIB = 1 << 20
CAP = {
    "manifest.json": 4 * MIB,
    "cone-fdk.exe": 16 * MIB,
    "kernels.ptx": 16 * MIB,
    "body-witness.json": MIB,
    "material.grid": 4096,
    "material-loss.json": 32 * MIB,
    **{
        f"{r}/{n}": limit
        for r in ("inspected", "reference")
        for n, limit in [
            ("shells.f32", 256 * MIB),
            ("scanner.grid", 4096),
            ("material.wall", 32 * MIB),
        ]
    },
}


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def need(b, why):
    if not b:
        raise ValueError(why)


for name, pin in SOURCE.items():
    need(sha(SCRIPTS / name) == pin, "unrecognized frozen source: " + name)
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))
import edge_review_packed as old  # noqa: E402


def object_path(store, pin):
    need(old.valid_pin(pin), "external SHA-256 required")
    return store / pin[:2] / pin


def fdpath(fd):
    return Path(f"/proc/self/fd/{fd}")


def native_run(program, ptx, atlas, grid, output, levels, budget):
    fds = (program, ptx, atlas, grid)
    cmd = [
        str(fdpath(program)),
        "--ptx",
        str(fdpath(ptx)),
        "--input",
        str(fdpath(atlas)),
        "--cylinder",
        str(fdpath(grid)),
        "--output",
        str(output),
        "--budget",
        str(budget),
    ]
    for v in levels:
        cmd += ["--level", str(v)]
    result = subprocess.run(cmd, pass_fds=fds, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"native exit {result.returncode}: {result.stderr.decode(errors='replace').strip()}"
        )
    return json.loads(result.stdout)


def produce(store, parent_pin, dest, levels, budget=16):
    need(old.valid_pin(parent_pin), "external SHA-256 required")
    need(type(budget) is int and 1 <= budget <= 65536, "invalid device budget")
    levels = list(levels)
    need(
        2 <= len(levels) <= 16
        and all(type(x) in (float, int) and math.isfinite(x) and x > 0 for x in levels),
        "invalid levels",
    )
    store = Path(store).resolve(strict=True)
    dest = fresh(dest)
    need(not dest.is_relative_to(store), "write outside archive")
    with staging(dest, ".witness-wall-") as tmp, ExitStack() as held:
        scratch = Path(tmp)
        view = scratch / "parent"
        view.mkdir()
        out = scratch / "review"
        out.mkdir()
        pins = {}
        fds = {}

        def bind(key, path, pin):
            need(key in CAP and key not in pins, "unexpected parent member")
            fd = held.enter_context(snapshot(object_path(store, pin), pin, limit=CAP[key]))
            pins[key] = pin
            fds[key] = fd
            if key not in ("cone-fdk.exe", "kernels.ptx"):
                need(path == view / key, "unexpected source path")
                path.parent.mkdir(exist_ok=True)
                path.symlink_to(fdpath(fd))

        projections, levels, nominal, gate, grid, classes, totals = old.prepare(
            view, bind, levels, parent_pin
        )
        w, h, *_ = grid
        need(2 * len(levels) * w * h * 32 <= 256 * MIB, "wall set exceeds 256 MiB output ceiling")
        exe, ptx = [
            held.enter_context(
                snapshot(BUILD / n, p, limit=16 * MIB, executable=n == "wall-pack")
            )
            for n, p in NATIVE.items()
        ]
        extra = [v for v in levels if v != nominal]
        shutil.copyfile(view / "material.grid", out / "material.grid")
        paths = []
        packs = {}
        stages = {}
        for role in old.ROLES:
            temp = scratch / role
            temp.mkdir()
            pack = temp / "levels.wallpack"
            prefix = role + "/"
            ak = prefix + "shells.f32"
            gk = prefix + "scanner.grid"
            nk = prefix + "material.wall"
            meta = native_run(exe, ptx, fds[ak], fds[gk], pack, extra, budget)
            n = old.check_stats(meta, view / gk, view / ak, pack, extra, budget)
            need(meta["gpu"] == "NVIDIA GB10", "unqualified CUDA device")
            digests = old.slices(pack, n, len(extra), temp)
            packs[role] = {"pack_sha256": sha(pack), "native": meta}
            stages[role] = []
            maps = []
            for i, v in enumerate(levels):
                p = out / f"{role}-{i:02}.wall"
                if v == nominal:
                    shutil.copyfile(view / nk, p)
                    expected = pins[nk]
                    stage = {"source": "parent"}
                else:
                    j = extra.index(v)
                    (temp / f"level-{j:02}.wall").rename(p)
                    expected = digests[j]
                    hist = old.status_hist(p)
                    need(hist == meta["status_counts_by_level"][j], "status histogram")
                    stage = {"source": "cuda_wall_pack", "pack_index": j, "status_counts": hist}
                need(sha(p) == expected, "wall digest")
                maps.append(p)
                stages[role].append(stage)
            paths.append(maps)
        result = old.reduce(paths, out, grid, levels.index(nominal), gate, classes)
        need(
            [sum(row.values()) for row in result["nominal_by_class"].values()] == totals,
            "nominal coverage",
        )
        outputs = {p.name: sha(p) for p in sorted(out.iterdir())}
        need(outputs["material.grid"] == pins["material.grid"], "copied material grid mismatch")
        doc = dict(
            format="CFDK-edge-review-v1",
            producer_schema="CFDK-Spark-witnessed-wall-v1",
            backend="sealed-GB10-wall-pack-v1",
            parent_sha256=parent_pin,
            parent_inputs_sha256=pins,
            parent_projection_sha256=projections,
            executable_products_sha256=NATIVE,
            frozen_source_sha256=SOURCE,
            adapter_sha256=sha(Path(__file__)),
            outputs_sha256=outputs,
            shape_ha=[h, w],
            levels_per_mm=levels,
            nominal_level_per_mm=nominal,
            loss_gate_mm=gate,
            class_codes=list(old.CLASSES),
            independent_pairs=len(levels) ** 2,
            additional_projection_passes=0,
            native_wall_calls=0,
            native_wall_pack_calls=2,
            native_budget_mib=budget,
            packs=packs,
            stages=stages,
            **result,
        )
        old.write_doc(out / "manifest.json", doc)
        publish(out, dest)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("store", type=Path)
    p.add_argument("parent_pin")
    p.add_argument("dest", type=Path)
    p.add_argument("--levels", nargs="+", required=True, type=float)
    p.add_argument("--budget", type=int, default=16)
    try:
        doc = produce(**vars(p.parse_args()))
        print(json.dumps(doc["counts"]))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as err:
        p.exit(1, f"witnessed-wall refused: {err}\n")
    raise SystemExit(2 if doc["counts"]["unknown"] else 0)
