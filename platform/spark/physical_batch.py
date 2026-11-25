"""Census of explicitly submitted physical residual qualifications. Never impute a missing case."""

import argparse
import json
import math
from pathlib import Path

import qualify_residual as qr
import residual_witness as rw

PLAN = "e8e263d1c02ab7782bed459d1195c528b35ef3fa7f1cdb07d22fa8fc576b4ed5"
KERNEL_REVIEW = "2f21f3c7a2d517b5580567cd2feae38d5090a7958f420d67a92a2f0c735f350c"
WITNESS = "b127c8be1a94e96a24e9a1404e5923a7bfdd8b2a7cf034db4e6ee76f4d6bafef"
need = rw.need


class Incomplete(ValueError):
    """No final result may be published for this submission."""


def same(a, b):
    return json.dumps(a, allow_nan=False, sort_keys=True) == json.dumps(
        b, allow_nan=False, sort_keys=True
    )


def totals(planes, shape):
    nx, ny, nz = shape
    n = nx * ny
    need(len(planes) == nz, "plane coverage")
    rows = []
    for z, p in enumerate(planes):
        need(type(p["z"]) is int and p["z"] == z, "plane identity")
        rw.pin(p["candidate_sha256"])
        rw.pin(p["reference_sha256"])
        row = p["gpu"]
        need(
            type(row) is list
            and len(row) == 12
            and all(type(row[i]) is int for i in (0, 1, 2, 3, 4, 5, 7)),
            "counter type",
        )
        need(
            row[0] == n
            and 0 <= row[1] <= n
            and n - row[1] >= row[2] >= row[3] >= row[4] >= row[5] >= 0
            and 0 <= row[7] < n,
            "plane counters or coordinates",
        )
        need(
            all(type(row[i]) in (int, float) and math.isfinite(row[i]) for i in (6, 8, 9, 10, 11))
            and min(row[6], row[9], row[10], row[11]) >= 0,
            "plane moments",
        )
        need(
            all((row[k + 2] > 0) == (row[6] > cut) for k, cut in enumerate(rw.CUTS))
            and (row[1] == n) == (row[6] == 0),
            "peak/counter disagreement",
        )
        rows.append(row)
    count = n * nz
    peak = max(r[6] for r in rows)
    z = next(i for i, r in enumerate(rows) if r[6] == peak)
    y, x = divmod(rows[z][7], nx)
    sums = [math.fsum(r[i] for r in rows) for i in (8, 9, 10, 11)]
    return dict(
        voxels=count,
        numeric_equal=sum(r[1] for r in rows),
        strict_exceedance=[sum(r[i] for r in rows) for i in range(2, 6)],
        max_abs=peak,
        first_maximum_xyz=[x, y, z],
        signed_mean=sums[0] / count,
        mean_abs=sums[1] / count,
        rms=math.sqrt(sums[2] / count),
        rel_l2=None if sums[3] == 0 else math.sqrt(sums[2] / sums[3]),
    )


def assess(kind, path, expected, pine_parent):
    spec = qr.volumes(kind, qr.ROOT.parent, pine_parent)
    spec.pop("paths")
    report = path / "qualification.json"
    q = qr.json_checked(report, expected)
    need(
        q["format"] == "CFDK-paired-volume-qualification-v1"
        and q["harness_sha256"] == KERNEL_REVIEW == qr.sha(qr.HERE / "qualify_residual.py")
        and q["witness_tool_sha256"] == WITNESS == qr.sha(qr.HERE / "residual_witness.py")
        and q["protocol_sha256"] == qr.RULES
        and q["build_sha256"] == qr.BUILD
        and q["row_fields"] == rw.FIELDS
        and same(q["source"], spec),
        "unqualified source or code lineage",
    )
    shape = spec["shape"]
    nz = shape[2]
    need(
        same(q["shape_xyz"], shape)
        and type(q["slices_verified_by_numpy"]) is int
        and q["slices_verified_by_numpy"] == nz,
        "shape or oracle coverage",
    )
    aggregate = totals(q["planes"], shape)
    need(
        same(aggregate, q["aggregate"])
        and aggregate["max_abs"] == spec["old"]["max_abs"]
        and aggregate["rel_l2"] is not None
        and math.isclose(aggregate["rel_l2"], spec["old"]["rel_l2"], rel_tol=2e-13, abs_tol=0),
        "aggregate or original baseline mismatch",
    )
    need(len(q["accounting"]) == 2, "repeat count")
    rows = [p["gpu"] for p in q["planes"]]
    for acc, b in zip(q["accounting"], (7, 1), strict=True):
        qr.accounts(dict(acc, rows=rows), shape, b, 32)
    z = aggregate["first_maximum_xyz"][2]
    witness = rw.verify(
        report, expected, z, path / "candidate-slice.f32", path / "reference-slice.f32"
    )
    need(
        witness["maximum_xyz"] == aggregate["first_maximum_xyz"]
        and witness["max_abs"] == aggregate["max_abs"],
        "witness mismatch",
    )
    return dict(
        qualification_sha256=expected,
        source_parents=spec["parents"],
        volume_sha256=spec["pins"],
        shape_xyz=shape,
        aggregate=aggregate,
        remeasured_witness=witness,
    )


def census(submissions, pine_parent, final=False):
    plan = qr.json_checked(qr.ROOT / "fixtures/physical-residual-batch-v1.json", PLAN)
    names = {c["case"] for c in plan["cases"]}
    jobs = {}
    pins = set()
    for kind, folder, expected in submissions:
        rw.pin(expected)
        need(
            kind in names and kind not in jobs and expected not in pins,
            "duplicate or unknown case",
        )
        jobs[kind] = (Path(folder), expected)
        pins.add(expected)
    has_pine = bool(set(jobs) & {"pine90", "pine720"})
    if has_pine:
        rw.pin(pine_parent)
    else:
        need(pine_parent is None, "unused pine parent pin")
    rows = []
    accepted = set()
    objects = {c["specimen"] for c in plan["cases"]}
    for c in plan["cases"]:
        kind = c["case"]
        if kind in jobs:
            data = assess(kind, *jobs[kind], pine_parent if kind.startswith("pine") else None)
            rows.append(dict(**c, status="verified", evidence=data))
            accepted.add(kind)
        else:
            rows.append(dict(**c, status="not_submitted"))
    if final and accepted != names:
        raise Incomplete("missing verified case(s): " + ", ".join(sorted(names - accepted)))
    full = sum(
        all(c["case"] in accepted for c in plan["cases"] if c["specimen"] == s) for s in objects
    )
    return dict(
        format="CFDK-physical-residual-batch-" + ("complete-v1" if final else "coverage-v1"),
        plan_sha256=PLAN,
        required_cases=len(names),
        verified_cases=len(accepted),
        required_specimens=len(objects),
        specimens_with_evidence=len({c["specimen"] for c in rows if c["status"] == "verified"}),
        specimens_with_all_planned_cases=full,
        all_cases_verified=accepted == names,
        cases=rows,
        scope="Same-grid numerical comparison on measured projections. Not-submitted cases have no metrics. View densities of one specimen are not independent specimens; no pooled accuracy, measurement repeatability or defect claim.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["status", "complete"])
    p.add_argument(
        "--case",
        action="append",
        nargs=3,
        default=[],
        metavar=("ID", "REPORT_DIR", "EXPECTED_SHA256"),
    )
    p.add_argument("--pine-parent")
    a = p.parse_args()
    try:
        print(
            json.dumps(
                census(a.case, a.pine_parent, final=a.mode == "complete"),
                indent=2,
                allow_nan=False,
            )
        )
    except Incomplete as e:
        p.exit(3, f"batch incomplete: {e}\n")
    except (ValueError, KeyError, TypeError, IndexError, OSError, OverflowError) as e:
        p.exit(1, f"batch refused: {e}\n")


if __name__ == "__main__":
    main()
