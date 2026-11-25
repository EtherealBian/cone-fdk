"""Qualify an edge review against raw shell profiles and explicit Cartesian pairs on CPU."""

import argparse
import zipfile
from pathlib import Path

import numpy as np
from common import REPORT
from compare_native import sha
from edge_figure import draw
from edge_probe import probe
from edge_review import CLASSES, ROLES, run, sampling
from loss_phantom import RELIEF, design
from remount_loss import write_doc
from validate_roi import require
from wall import WALL_DTYPE

LEVELS = (0.018, 0.020, 0.022)


def raw_wall(x, level, radii):
    # x: FP64 radial profiles as columns. Determine missing/ambiguous edges before interpolating.
    mask = x > level
    rising = ~mask[:-1] & mask[1:]
    falling = mask[:-1] & ~mask[1:]
    knot = x[1:-1] == level
    lo = np.minimum(x[:-2], x[2:])
    hi = np.maximum(x[:-2], x[2:])
    status = np.select(
        [
            ~np.isfinite(x).all(0),
            (x[0] >= level) | (x[-1] >= level),
            (knot & ~((lo < level) & (hi > level))).any(0),
            rising.sum(0) > 1,
            ~rising.any(0),
            mask.sum(0) < 2,
        ],
        [5, 2, 4, 3, 1, 6],
        default=0,
    )
    keep = np.flatnonzero(status == 0)
    t = np.full(len(status), np.nan)

    def intersect(transitions):
        k = transitions[:, keep].argmax(0)
        a = x[k, keep]
        b = x[k + 1, keep]
        # Interpolate scanner radii independently of CUDA's fractional-index thickness.
        return radii[k] + (radii[k + 1] - radii[k]) * (level - a) / (b - a)

    t[keep] = intersect(falling) - intersect(rising)
    return status, t


def raw_oracle(parent, review, doc):
    w, h, d, r0, dr, _ = sampling(parent / "material.grid")
    levels = doc["levels_per_mm"]
    gate = doc["loss_gate_mm"]
    radii = r0 + np.arange(d) * dr
    atlas = [np.memmap(parent / r / "shells.f32", "<f4", "r", shape=(d, h * w)) for r in ROLES]
    walls = [
        [
            np.memmap(review / f"{r}-{i:02}.wall", WALL_DTYPE, "r", shape=(h * w,))
            for i in range(len(levels))
        ]
        for r in ROLES
    ]
    actual = np.fromfile(review / "class.u8", "u1")
    ends = [np.fromfile(review / n, "<f8") for n in ["min-loss.f64", "max-loss.f64"]]
    maxerr = 0.0
    blind = 0
    for start in range(0, h * w, 512):
        sl = slice(start, min(start + 512, h * w))
        inputs = []
        for data, readers in zip(atlas, walls, strict=True):
            x = data[:, sl].astype("f8")
            group = []
            for level, reader in zip(levels, readers, strict=True):
                s, t = raw_wall(x, level, radii)
                rec = reader[sl]
                np.testing.assert_array_equal(s, rec["status"])
                err = np.abs(t - rec["thickness"])
                maxerr = max(maxerr, float(np.max(err[np.isfinite(err)], initial=0)))
                np.testing.assert_allclose(t, rec["thickness"], rtol=0, atol=1e-12)
                group.append(t)
            inputs.append(group)
        pairs = np.stack([a - b for a in inputs[0] for b in inputs[1]])
        votes = pairs >= gate
        unknown = ~np.isfinite(pairs).all(0)
        cls = np.where(unknown, 3, np.where(votes.all(0), 2, np.where(votes.any(0), 1, 0)))
        np.testing.assert_array_equal(cls, actual[sl])
        diag = votes[:: len(levels) + 1]
        blind += int(((cls == 1) & (diag.all(0) == diag.any(0))).sum())
        for arr, calc in zip(ends, [pairs.min(0), pairs.max(0)], strict=True):
            np.testing.assert_allclose(arr[sl], calc, rtol=0, atol=2e-12)
    require(blind == doc["sensitive_missed_by_same_level"], "same-level comparison mismatch")
    return dict(
        raw_wall_records_checked=2 * len(levels) * w * h,
        explicit_pair_cells=len(levels) ** 2 * w * h,
        max_abs_cpu_cuda_thickness_mm=maxerr,
    )


def pack(job, dest):
    with zipfile.ZipFile(dest, "w") as z:
        for p in sorted(job.iterdir()):
            info = zipfile.ZipInfo(p.name)
            info.create_system = 0
            info.external_attr = 0x20
            z.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def qualify(parents, work, dest):
    work.mkdir(parents=True, exist_ok=False)
    dest.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, src, levels in [
        ("repeat", "repeat", LEVELS),
        ("noisy-change", "noisy-change", LEVELS),
        ("invalid-edge-control", "repeat", (0.020, 0.040)),
    ]:
        parent = parents / src
        job = work / name
        digest = sha(parent / "manifest.json")
        doc = run(parent, job, levels, parent_sha256=digest)
        c = doc["counts"]
        oracle = raw_oracle(parent, job, doc)
        entry = cases[name] = dict(
            parent_manifest_sha256=digest,
            review_manifest_sha256=sha(job / "manifest.json"),
            levels_per_mm=levels,
            loss_gate_mm=doc["loss_gate_mm"],
            counts=c,
            nominal_by_class=doc["nominal_by_class"],
            sensitive_missed_by_same_level=doc["sensitive_missed_by_same_level"],
            native_wall_calls=doc["native_wall_calls"],
            additional_projection_passes=doc["additional_projection_passes"],
            **oracle,
        )
        if name == "repeat":
            require(c["always_below"] == 14760, "unchanged sample did not stay below gate")
        elif name == "invalid-edge-control":
            require(
                c["unknown"] == 14760 and probe(job, 20, 134)["classification"] == "unknown",
                "invalid edge became a decision",
            )
        else:
            require(
                c["always_flagged"] > 0
                and c["sensitive"] > 0
                and c["unknown"] == 0
                and doc["sensitive_missed_by_same_level"] > 0,
                "no independent-pair sensitivity in known relief",
            )
            # Geometric design is not an input to the review; keep it separate from CPU/CUDA agreement.
            truth = design(None, RELIEF)[1] >= doc["loss_gate_mm"]
            cls = np.fromfile(job / "class.u8", "u1").reshape(truth.shape)
            entry["design_cross_tab"] = {
                k: dict(zip(CLASSES, np.bincount(cls[select], minlength=4).tolist(), strict=True))
                for k, select in [("below", ~truth), ("flagged", truth)]
            }
            iou = float(
                np.count_nonzero((cls == 2) & truth) / np.count_nonzero((cls == 2) | truth)
            )
            require(iou >= 0.90, "stable footprint versus analytic relief")
            entry["always_flagged_iou_vs_design"] = iou
            cell = probe(job, 23, 122)
            pairs = cell["pairs"]
            require(
                cell["classification"] == "sensitive" and all(p["flagged"] for p in pairs[::4]),
                "example no longer straddles the gate",
            )
            write_doc(dest / "edge-cell.json", cell)
            pack(job, dest / "edge-review-example.zip")
            draw(job, cell, dest / "edge-review.png")
            entry["example_cell"] = [cell["row"], cell["column"]]
    for name in ("slide", "missing-reference", "noisy-change-fixture-only"):
        try:
            run(parents / name, work / name, LEVELS)
        except ValueError as e:
            require(
                "not an accepted witness" in str(e) and not (work / name).exists(),
                "incorrect rejection",
            )
        else:
            raise ValueError("rejected or unguarded parent reached edge review")
    write_doc(
        dest / "edge-qualification.json",
        dict(
            protocol="Finite level scenarios on independently projected simulated remounts. CPU reconstructs crossings from the FP32 shells and enumerates all pairs; no physical uncertainty or accuracy bound.",
            cases=cases,
            rejected_parents=["slide", "missing-reference", "noisy-change-fixture-only"],
            evidence_sha256={
                n: sha(dest / n)
                for n in ["edge-cell.json", "edge-review-example.zip", "edge-review.png"]
            },
        ),
    )
    print(
        "PASS: raw profiles, explicit level pairs, negative controls and portable wall evidence",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parents", type=Path)
    p.add_argument("work", type=Path)
    p.add_argument("--reports", dest="dest", type=Path, default=REPORT)
    qualify(**vars(p.parse_args()))
