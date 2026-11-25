"""Recompute a witnessed edge review with one non-nominal CUDA wall pack per scan."""

import argparse
import json
import math
import re
import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from axis_gauge import pinned_reader
from common import BUILD
from compare_native import sha
from edge_review import (
    CLASSES,
    MAPS,
    ROLES,
    check,
    nominal_classes,
    reduce,
    sampling,
    scenarios,
    witness_gate,
)
from remount_loss import write_doc
from wall import STATUS, WALL_DTYPE
from wall_pack import check_stats, measure_pack, slices


def valid_pin(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def prepare(parent, bind, levels, expected):
    """Validate the witnessed parent before invoking a native backend; no derived child is trusted."""
    mp = parent / "manifest.json"
    bind("manifest.json", mp, expected)
    check(mp.stat().st_size <= 4 << 20, "oversize parent manifest")
    doc = json.loads(mp.read_bytes())
    check(
        doc["format"] == "CFDK-remount-witness-job-v1"
        and doc["status"] in ("compared", "incomplete_coverage"),
        "parent is not an accepted witness measurement",
    )
    src, outputs = doc["inputs_sha256"], doc["outputs_sha256"]
    projections = {r: src[r + ".f32"] for r in ROLES}
    check(
        all(map(valid_pin, projections.values())) and len(set(projections.values())) == 2,
        "missing or identical parent projection fingerprints",
    )
    for name in ("cone-fdk.exe", "kernels.ptx"):
        bind(name, BUILD / name, src[name])
    for name in (
        "body-witness.json",
        "material.grid",
        "material-loss.json",
        *(r + "/" + n for r in ROLES for n in ("scanner.grid", "shells.f32", "material.wall")),
    ):
        bind(name, parent / name, outputs[name])
    check(
        all(src[n] == outputs[n] for n in ("material.grid", "body-witness.json")),
        "copied calibration mismatch",
    )
    check((parent / "body-witness.json").stat().st_size <= 1 << 20, "oversize calibration")
    witness_gate(doc, json.loads((parent / "body-witness.json").read_bytes()))
    nominal, gate = (doc["parameters"][k] for k in ("level_per_mm", "loss_mm"))
    levels = scenarios(levels, nominal)
    check(math.isfinite(gate) and gate > 0, "invalid parent loss gate")
    grid = sampling(parent / "material.grid")
    w, h, d, *_ = grid
    for r in ROLES:
        folder = parent / r
        check(
            sampling(folder / "scanner.grid") == grid
            and (folder / "shells.f32").stat().st_size == 4 * w * h * d
            and (folder / "material.wall").stat().st_size == 32 * w * h,
            "parent grid or payload size mismatch",
        )
    lp = parent / "material-loss.json"
    check(lp.stat().st_size <= 32 << 20, "parent loss JSON exceeds the 32 MiB review limit")
    loss = json.loads(lp.read_bytes())
    cov = loss["coverage"]
    check(
        loss["format"] == "CFDK-loss-v1"
        and loss["threshold_mm"] == gate
        and (loss["columns"], loss["rows"]) == (w, h)
        and cov == doc["loss"]["coverage"]
        and doc["status"] == ("incomplete_coverage" if cov["unknown"] else "compared"),
        "parent loss mismatch",
    )
    classes = nominal_classes(w, h, loss["runs"])
    counts = np.bincount(classes, minlength=3).tolist()
    check(
        counts == [cov[k] for k in ("below_threshold", "flagged", "unknown")]
        and sum(counts[:2]) == cov["paired_valid"],
        "parent nominal coverage mismatch",
    )
    return projections, levels, nominal, gate, grid, classes, counts


def status_hist(path):
    counts = np.zeros(len(STATUS), dtype="i8")
    with path.open("rb") as f:
        while (rows := np.fromfile(f, WALL_DTYPE, 32768)).size:
            values = rows["status"]
            check((values < len(STATUS)).all(), "unknown wall status code")
            counts += np.bincount(values, minlength=len(STATUS))
    return counts.tolist()


def run(parent_job, job, levels, *, parent_sha256, companion_sha256, budget=16):
    parent, job = map(lambda p: Path(p).resolve(), (parent_job, job))
    if job.exists():
        raise FileExistsError(job)
    check(not job.is_relative_to(parent), "write outside parent")
    check(type(budget) is int and 1 <= budget <= 65536, "invalid native budget")
    check(
        valid_pin(parent_sha256) and valid_pin(companion_sha256),
        "two external SHA-256 pins required",
    )
    with ExitStack() as held:
        pins = {}

        def bind(key, path, expected):
            check(valid_pin(expected), "missing digest: " + key)
            held.enter_context(pinned_reader(path))
            check(sha(path) == expected, "SHA-256 mismatch: " + key)
            pins[key] = expected

        projections, levels, nominal, gate, grid, classes, totals = prepare(
            parent, bind, levels, parent_sha256
        )
        bind("wall-pack.exe", BUILD / "wall-pack.exe", companion_sha256)
        w, h, *_ = grid
        extra = [v for v in levels if v != nominal]
        job.parent.mkdir(parents=True, exist_ok=True)
        # A failure after either native call cannot publish a half review. Same-volume rename.
        with TemporaryDirectory(dir=job.parent, prefix=".packed-") as tmp:
            scratch = Path(tmp)
            out = scratch / "review"
            out.mkdir()
            stages, native, hashes, paths = {}, {}, {}, []
            with ExitStack() as generated:

                def seal(name, expected=None):
                    dest = out / name
                    generated.enter_context(pinned_reader(dest))
                    digest = hashes[name] = sha(dest)
                    check(
                        expected is None or digest == expected, "wall/grid copy mismatch: " + name
                    )

                shutil.copyfile(parent / "material.grid", out / "material.grid")
                seal("material.grid", pins["material.grid"])
                for role in ROLES:
                    folder = parent / role
                    staging = scratch / role
                    staging.mkdir()
                    pack = staging / "levels.wallpack"
                    atlas, spec = folder / "shells.f32", folder / "scanner.grid"
                    meta = measure_pack(atlas, spec, pack, extra, budget=budget)
                    with pinned_reader(pack):
                        size = check_stats(meta, spec, atlas, pack, extra, budget)
                        digests = slices(pack, size, len(extra), staging)
                        native[role] = dict(pack_sha256=sha(pack), native=meta)
                    stages[role] = []
                    maps = []
                    for i, lvl in enumerate(levels):
                        name = f"{role}-{i:02}.wall"
                        dest = out / name
                        if lvl == nominal:
                            shutil.copyfile(folder / "material.wall", dest)
                            seal(name, pins[role + "/material.wall"])
                            stage = dict(source="parent")
                        else:
                            j = extra.index(lvl)
                            (staging / f"level-{j:02}.wall").rename(dest)
                            seal(name, digests[j])
                            counts = status_hist(dest)
                            check(
                                counts == meta["status_counts_by_level"][j],
                                "packed status histogram mismatch",
                            )
                            stage = dict(
                                source="cuda_wall_pack", pack_index=j, status_counts=counts
                            )
                        stages[role].append(stage)
                        maps.append(dest)
                    paths.append(maps)
                summary = reduce(paths, out, grid, levels.index(nominal), gate, classes)
                check(
                    [sum(row.values()) for row in summary["nominal_by_class"].values()] == totals,
                    "nominal coverage disagrees with parent",
                )
                for name in MAPS:
                    seal(name)
                log = dict(
                    format="CFDK-edge-review-v1",
                    backend="witness-wall-pack-v1",
                    shape_ha=[h, w],
                    levels_per_mm=levels,
                    nominal_level_per_mm=nominal,
                    loss_gate_mm=gate,
                    class_codes=list(CLASSES),
                    independent_pairs=len(levels) ** 2,
                    additional_projection_passes=0,
                    native_wall_calls=0,
                    native_wall_pack_calls=len(native),
                    native_budget_mib=budget,
                    inputs_sha256=pins,
                    parent_projection_sha256=projections,
                    outputs_sha256=hashes,
                    software_sha256={
                        n: sha(Path(__file__).with_name(n))
                        for n in (
                            "edge_review_packed.py",
                            "edge_review.py",
                            "wall_pack.py",
                            "wall.py",
                        )
                    },
                    stages=stages,
                    packs=native,
                    **summary,
                )
                write_doc(out / "manifest.json", log)
            out.rename(job)
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent_job", type=Path)
    p.add_argument("job", type=Path)
    p.add_argument("--levels", nargs="+", type=float, required=True)
    p.add_argument("--budget", type=int, default=16)
    for n in ("parent", "companion"):
        p.add_argument("--" + n + "-sha256", required=True)
    try:
        doc = run(**vars(p.parse_args()))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as e:
        p.exit(1, f"packed edge review failed: {e}\n")
    print(json.dumps(doc["counts"]))
    raise SystemExit(2 if doc["counts"]["unknown"] else 0)
