"""Audit radial loss decisions across independently selected edge levels in a witness-checked job."""

import argparse
import json
import math
import re
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from axis_gauge import pinned_reader
from common import BUILD
from compare_native import sha
from remount_loss import write_doc
from wall import STATUS, WALL_DTYPE, measure

ROLES = ("reference", "inspected")
CLASSES = ("always_below", "sensitive", "always_flagged", "unknown")
MAPS = ("class.u8", "min-loss.f64", "max-loss.f64")


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def scenarios(levels, nominal):
    levels = sorted(levels)
    check(
        2 <= len(levels) <= 16
        and all(math.isfinite(x) and x > 0 for x in levels)
        and len(set(levels)) == len(levels)
        and nominal in levels,
        "supply 2..16 distinct positive finite levels including the parent nominal level",
    )
    return levels


def witness_gate(parent, body):
    limit = body["residual_limit_mm"]
    check(
        body["format"] == "CFDK-body-witness-v1"
        and len(body["material_xyz_mm"]) == 3
        and math.isfinite(limit)
        and limit > 0,
        "invalid witness calibration",
    )
    for role in ROLES:
        rec = parent["scans"][role]["body_witnesses"]
        check(
            [p["id"] for p in rec] == [1, 2, 3]
            and all(
                p["status"] == "pass"
                and p["residual_mm"] is not None
                and 0 <= p["residual_mm"] <= limit
                for p in rec
            ),
            "missing or failed body witness: " + role,
        )


def sampling(path):
    tok = path.read_text(encoding="utf-8-sig").split()
    check(len(tok) == 16 and tok[0] == "CFDKCYL1", "bad cylinder header")
    w, h, d = map(int, tok[1:4])
    r0, dr, dz = map(float, tok[-3:])
    check(
        3 <= w <= 4096
        and 1 <= h <= 4096
        and 3 <= d <= 4096
        and all(math.isfinite(x) and x > 0 for x in (r0, dr, dz)),
        "bad sampling",
    )
    return w, h, d, r0, dr, dz


def thickness(arr, lo, hi):
    """Validate native records before reducing them; an invalid file is not an unknown ray."""
    t, i, o, s = (arr[k] for k in ("thickness", "inner", "outer", "status"))
    ok = s == 0
    tol = 16 * np.finfo(float).eps
    with np.errstate(invalid="ignore", over="ignore"):
        valid = (
            np.isfinite(t)
            & np.isfinite(i)
            & np.isfinite(o)
            & (i > 0)
            & (t > 0)
            & (o > i)
            & (i >= lo - tol * lo)
            & (o <= hi + tol * hi)
            & (abs((o - i) - t) <= tol * o)
        )
    check(
        (
            (arr["reserved"] == 0)
            & (s < len(STATUS))
            & np.where(ok, valid, np.isnan(i) & np.isnan(o) & np.isnan(t))
        ).all(),
        "malformed wall record",
    )
    return t


def bounds(a, b, gate):
    # Shape: levels x cells. A rejected level is NaN: min/max (not nanmin/nanmax) propagate it.
    low = a.min(0) - b.max(0)
    high = a.max(0) - b.min(0)
    labels = np.select([np.isnan(low), low >= gate, high >= gate], [3, 2, 1], default=0).astype(
        "u1"
    )
    diag = a - b >= gate
    missed = (labels == 1) & (diag.all(0) == diag.any(0))
    return labels, low, high, missed


def nominal_classes(w, h, runs):
    """Native loss runs are ordered within rows; decode just the decision, not the region ID."""
    tag = np.zeros(w * h, dtype="u1")
    end = 0
    for y, x, length, ident in runs:
        start = y * w + x
        check(
            0 <= y < h
            and 0 <= x < w
            and 0 < length <= w - x
            and start >= end
            and (0 < ident <= w * h or -50 <= ident <= -2),
            "invalid nominal loss run",
        )
        end = start + length
        tag[start:end] = 1 if ident > 0 else 2
    return tag


def reduce(paths, job, grid, nominal_idx, gate, parent_classes, chunk=4096):
    w, h, d, r0, dr, _ = grid
    counts = np.zeros((3, 4), dtype="i8")  # nominal below / flagged / unknown x sensitivity class
    missed = 0
    with ExitStack() as stack:
        handles = [[stack.enter_context(p.open("rb")) for p in ps] for ps in paths]
        outs = [stack.enter_context((job / n).open("xb")) for n in MAPS]
        for pos in range(0, w * h, chunk):
            n = min(w * h - pos, chunk)
            sets = []
            for files in handles:
                rows = []
                for f in files:
                    arr = np.fromfile(f, WALL_DTYPE, n)
                    check(len(arr) == n, "truncated wall map")
                    rows.append(thickness(arr, r0, r0 + (d - 1) * dr))
                sets.append(np.stack(rows))
            a, b = sets
            labels, low, high, blind = bounds(a, b, gate)
            delta = a[nominal_idx] - b[nominal_idx]
            nominal = np.where(np.isnan(delta), 2, (delta >= gate).astype(int))
            check(
                np.array_equal(nominal, parent_classes[pos : pos + n]),
                "nominal cell disagrees with parent loss run",
            )
            counts += np.bincount(4 * nominal + labels, minlength=12).reshape(3, 4)
            missed += int(blind.sum())
            for f, arr in zip(
                outs,
                (labels, low.astype("<f8", copy=False), high.astype("<f8", copy=False)),
                strict=True,
            ):
                arr.tofile(f)
        check(all(not f.read(1) for group in handles for f in group), "oversized wall map")
    return dict(
        counts=dict(zip(CLASSES, counts.sum(0).tolist(), strict=True)),
        nominal_by_class={
            k: dict(zip(CLASSES, row.tolist(), strict=True))
            for k, row in zip(("below", "flagged", "unknown"), counts, strict=True)
        },
        sensitive_missed_by_same_level=missed,
    )


def run(parent_job, job, levels, *, budget=16, parent_sha256=None):
    parent_job, job = Path(parent_job).resolve(), Path(job).resolve()
    check(not job.is_relative_to(parent_job), "write the review outside its parent job")
    check(1 <= budget <= 65536, "invalid native wall budget")
    with ExitStack() as held:
        input_hash = {}

        def bind(key, path, expected=None):
            held.enter_context(pinned_reader(path))
            digest = sha(path)
            if expected is not None:
                check(
                    isinstance(expected, str) and digest == expected, "SHA-256 mismatch: " + key
                )
            input_hash[key] = digest
            return digest

        mp = parent_job / "manifest.json"
        bind("manifest.json", mp, parent_sha256)
        parent = json.loads(mp.read_bytes())
        check(
            parent["format"] == "CFDK-remount-witness-job-v1"
            and parent["status"] in ("compared", "incomplete_coverage"),
            "parent is not an accepted witness measurement",
        )
        sources = parent["inputs_sha256"]
        artifacts = parent["outputs_sha256"]
        # An external digest may anchor the parent; internal hashes alone do not authenticate it.
        projections = {r: sources[r + ".f32"] for r in ROLES}
        check(
            all(
                isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)
                for v in projections.values()
            )
            and len(set(projections.values())) == 2,
            "missing or identical parent projection fingerprints",
        )
        for name in ("cone-fdk.exe", "kernels.ptx"):
            check(isinstance(sources[name], str), "missing native tool digest")
            bind(name, BUILD / name, sources[name])
        files = [
            "body-witness.json",
            "material.grid",
            "material-loss.json",
            *(
                r + "/" + s
                for r in ROLES
                for s in ("scanner.grid", "material.wall", "shells.f32")
            ),
        ]
        for name in files:
            check(isinstance(artifacts[name], str), "missing parent artifact digest")
            bind(name, parent_job / name, artifacts[name])
        for name in ("material.grid", "body-witness.json"):
            check(sources[name] == input_hash[name], "copied calibration mismatch")
        witness_gate(parent, json.loads((parent_job / "body-witness.json").read_bytes()))
        nominal = parent["parameters"]["level_per_mm"]
        gate = parent["parameters"]["loss_mm"]
        levels = scenarios(levels, nominal)
        check(math.isfinite(gate) and gate > 0, "invalid parent loss gate")
        grid = sampling(parent_job / "material.grid")
        w, h, d, *_ = grid
        loss_path = parent_job / "material-loss.json"
        check(
            loss_path.stat().st_size <= 32 << 20,
            "parent loss JSON exceeds the 32 MiB review limit",
        )
        loss = json.loads(loss_path.read_bytes())
        cov = loss["coverage"]
        check(
            loss["format"] == "CFDK-loss-v1"
            and loss["threshold_mm"] == gate
            and (loss["columns"], loss["rows"]) == (w, h)
            and parent["loss"]["coverage"] == cov
            and parent["status"] == ("incomplete_coverage" if cov["unknown"] else "compared"),
            "parent loss mismatch",
        )
        parent_classes = nominal_classes(w, h, loss["runs"])
        del loss
        for r in ROLES:
            folder = parent_job / r
            check(
                sampling(folder / "scanner.grid") == grid
                and (folder / "shells.f32").stat().st_size == 4 * w * h * d
                and (folder / "material.wall").stat().st_size == 32 * w * h,
                "parent grid or payload size mismatch",
            )
        job.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(parent_job / "material.grid", job / "material.grid")
        paths = []
        stages = {}
        out_hash = {}
        for role in ROLES:
            folder = parent_job / role
            maps = []
            stages[role] = []
            for i, lvl in enumerate(levels):
                name = f"{role}-{i:02}.wall"
                dest = job / name
                if lvl == nominal:
                    shutil.copyfile(folder / "material.wall", dest)
                    stage = dict(source="parent")
                else:
                    m = measure(
                        folder / "shells.f32", folder / "scanner.grid", dest, lvl, budget=budget
                    )
                    check(
                        m["redzones_verified"]
                        and 0 < m["peak_owned_bytes"] <= m["budget_bytes"] == budget << 20
                        and m["shape_ha"] == [h, w]
                        and m["shells"] == d
                        and m["input_bytes"] == 4 * w * h * d
                        and m["output_bytes"] == dest.stat().st_size == 32 * w * h
                        and m["threshold"] == lvl,
                        "native wall accounting mismatch",
                    )
                    stage = dict(
                        source="cuda_wall",
                        gpu=m["gpu"],
                        peak_owned_bytes=m["peak_owned_bytes"],
                        status_counts=m["status_counts"],
                    )
                held.enter_context(pinned_reader(dest))
                maps.append(dest)
                out_hash[name] = sha(dest)
                if lvl == nominal:
                    check(
                        out_hash[name] == input_hash[role + "/material.wall"],
                        "nominal map copy mismatch",
                    )
                stages[role].append(stage)
            paths.append(maps)
        summary = reduce(paths, job, grid, levels.index(nominal), gate, parent_classes)
        totals = [sum(row.values()) for row in summary["nominal_by_class"].values()]
        check(
            totals == [cov["below_threshold"], cov["flagged"], cov["unknown"]]
            and sum(totals[:2]) == cov["paired_valid"],
            "nominal map disagrees with parent coverage",
        )
        for name in ("material.grid", *MAPS):
            held.enter_context(pinned_reader(job / name))
            out_hash[name] = sha(job / name)
        check(
            out_hash["material.grid"] == input_hash["material.grid"],
            "material grid copy mismatch",
        )
        log = dict(
            format="CFDK-edge-review-v1",
            shape_ha=[h, w],
            levels_per_mm=levels,
            nominal_level_per_mm=nominal,
            loss_gate_mm=gate,
            class_codes=list(CLASSES),
            independent_pairs=len(levels) ** 2,
            additional_projection_passes=0,
            native_wall_calls=sum(
                s["source"] == "cuda_wall" for scan in stages.values() for s in scan
            ),
            native_budget_mib=budget,
            inputs_sha256=input_hash,
            parent_projection_sha256=projections,
            outputs_sha256=out_hash,
            software_sha256={
                n: sha(Path(__file__).with_name(n)) for n in ("edge_review.py", "wall.py")
            },
            stages=stages,
            **summary,
        )
        temp = job / "manifest.partial.json"
        write_doc(temp, log)
        temp.replace(job / "manifest.json")
        return log


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parent_job", type=Path)
    p.add_argument("job", type=Path)
    p.add_argument("--levels", type=float, nargs="+", required=True)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument(
        "--parent-sha256",
        help="expected digest of the witness job manifest, from a trusted record",
    )
    try:
        result = run(**vars(p.parse_args()))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as e:
        p.exit(1, f"edge review failed: {e}\n")
    print(json.dumps(result["counts"]))
    raise SystemExit(2 if result["counts"]["unknown"] else 0)
