"""Independently account for the 28 archived GB10 trials, without CUDA or the benchmark driver."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from zipfile import ZipFile

RESULT = "e7d675a35bb055615717dc5dd16910f473738b67b84a228c98f9bfa763ca6563"
PROTOCOL = "0b865ba5a30363396ee8181ef77b0a47f0f8237d29f2b2febc3260719de6c658"
DRIVER = "7c3226dbfe8962a51474eb5fdeb30e981a3339709112ca5db97dd7aedba51676"
PARENT = "250443f4c5ddd9cb3458617a072b7f1c4d9c40b539e3bffaeb73f4d1b9ece740"
SCAN = "2417f08c8a91a95b061acfe1aa2126932b6882f8bd75673e65df261d2738a43b"
CASES = ("pine90", "pine720")


def need(ok, why):
    if not ok:
        raise ValueError(why)


def checked(raw, pin):
    need(hashlib.sha256(raw).hexdigest() == pin, "digest mismatch")
    return raw


def schedule():
    # Start at warm-up block -1; invert both the specimen order and its budget order on alternate blocks.
    rows = []
    for block in range(-1, 6):
        for case, first in (
            (("pine720", 64), ("pine90", 384))
            if block % 2
            else (("pine90", 64), ("pine720", 384))
        ):
            rows.extend([[block, case, first], [block, case, 448 - first]])
    return rows


def summarize(rows, order, parent, scan):
    need(
        order == schedule() and len(rows) == len(order) == 28, "incomplete or reordered schedule"
    )
    times = {}
    resources = {}
    for row, (block, case, budget) in zip(rows, order, strict=True):
        need(
            [row["block"], row["case"], row["budget_mib"]] == [block, case, budget],
            "trial order/identity",
        )
        p = parent["scans"][case]
        s = scan["scans"][case]
        r = row["run"]
        views = int(case[4:])
        height, width = s["shape_vu"]
        n = 256
        b = 16
        fft = 1 << (2 * width - 1).bit_length()
        cap = budget << 20
        plane = n * n * 4
        proj = views * height * width * 4
        need(
            row["sha256"] == p["dense_sha256"]
            and p["views"] == s["views"] == views
            and p["projections_sha256"] == s["projections_sha256"]
            and p["geometry_sha256"] == s["geom_sha256"]
            and s["projection_bytes"] == proj,
            "scan or volume mismatch",
        )
        # Planner reserves six guarded allocations; the zero-byte FFT work buffer is not allocated.
        payload = height * b * fft * 8 + height * width * b * 4 + fft * 4 + views * 8
        slab = min(n, (cap - payload - 6 * 512) // plane)
        passes = math.ceil(n / slab)
        batches = math.ceil(views / b) * passes
        peak = payload + 5 * 512 + slab * plane
        fields = dict(
            gpu="NVIDIA GB10",
            views=views,
            n=n,
            batch=b,
            requested_batch=b,
            voxel_spacing_mm=44 / n,
            planner="batch-first",
            budget_bytes=cap,
            effective_budget_bytes=cap,
            roi_origin_xyz=[0] * 3,
            output_shape_xyz=[n] * 3,
            output_bytes=n * plane,
            slab=slab,
            slabs=passes,
            projection_passes=passes,
            projection_bytes_read=passes * proj,
            filter_batches=batches,
            launches_bp=batches,
            peak_owned_bytes=peak,
            host_pinned_bytes=height * width * b * 4 + min(slab * n * n, 2 << 20) * 4,
            fft_workspace_bytes=0,
            feasible_plans_scored=[dict(batch=b, passes=passes)],
        )
        need(
            r["redzones_verified"] is True and all(r.get(k) == v for k, v in fields.items()),
            "native byte/plan accounting",
        )
        t = r["process_seconds"]
        need(type(t) in (int, float) and math.isfinite(t) and t > 0, "invalid timing")
        times[block, case, budget] = t
        resources[case, budget] = dict(
            slab_depth=slab,
            projection_passes=passes,
            projection_bytes_read=passes * proj,
            peak_owned_bytes=peak,
            filter_batches=batches,
        )
    out = {}
    for case in CASES:
        diffs = [times[i, case, 64] - times[i, case, 384] for i in range(6)]
        out[case] = {
            "by_budget_mib": {
                str(k): dict(
                    resources[case, k],
                    median_process_seconds=median(times[i, case, k] for i in range(6)),
                )
                for k in (64, 384)
            },
            "paired_seconds_64_minus_384": diffs,
            "median_paired_seconds_64_minus_384": median(diffs),
            "output_sha256": parent["scans"][case]["dense_sha256"],
        }
    return {
        "format": "CFDK-GB10-paired-pine-budget-review-v1",
        "raw_trials": 28,
        "excluded_warmups": 4,
        "measured_pairs_per_condition": 6,
        "cases": out,
        "scope": "One pine cone at two view counts. Native process time, not kernel time. Requested budget differs from peak owned allocation; neither is total system memory. No speed gate.",
    }


def review(archive, parents):
    need(archive.stat().st_size < 256 * 1024, "archive too large")
    names = [f"trial-{i:02d}.json" for i in range(28)]
    with ZipFile(archive) as z:
        need(
            sorted(z.namelist()) == sorted(names + ["driver.py", "result.json", "protocol.json"])
            and all(x.file_size <= 8192 for x in z.infolist()),
            "archive membership/size",
        )
        checked(z.read("driver.py"), DRIVER)
        result = json.loads(checked(z.read("result.json"), RESULT))
        proto = json.loads(checked(z.read("protocol.json"), PROTOCOL))
        need(
            proto["parent_sha256"] == PARENT
            and proto["scan_sha256"] == SCAN
            and proto["driver_sha256"] == DRIVER
            and result["protocol_sha256"] == PROTOCOL
            and result["measured_pairs_per_condition"] == 6,
            "protocol linkage",
        )
        rows = [
            json.loads(checked(z.read(name), pin))
            for name, pin in zip(names, result["row_sha256"], strict=True)
        ]
    parent = json.loads(checked((parents / "pine-parent.json").read_bytes(), PARENT))
    scan = json.loads(checked((parents / "pine-scan.json").read_bytes(), SCAN))
    return summarize(rows, proto["order"], parent, scan)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("physical_batch", type=Path)
    a = cli.parse_args()
    print(json.dumps(review(a.archive, a.physical_batch), indent=2, allow_nan=False))
