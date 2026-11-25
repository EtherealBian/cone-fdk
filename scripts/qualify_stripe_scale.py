"""Non-measurement scaling fixture, never accepted by the production edge-review entry point."""

import argparse
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from itertools import product
from pathlib import Path

import numpy as np
from compare_native import sha
from edge_review import CLASSES, MAPS, ROLES
from region_review import MAX_CELLS, labels
from remount_loss import write_doc
from stripe_review import scan
from validate_roi import require
from wall import WALL_DTYPE

WIDTH = 960
LEVELS = [0.018, 0.020, 0.022]
GATE = 0.5


def field(rows, h, w):
    y = np.asarray(rows)[:, None]
    x = np.arange(w)[None, :]
    lanes = (x <= 1) | (x == w - 1) | (x == w // 3)
    neck = (y == h // 2) & (x <= w // 3)
    islands = (x > w // 2) & (x < w - 5) & (y > h // 10) & (y < h // 3) & ((x + y) % 2 == 0)
    return np.where(lanes | islands, 0.875, np.where(neck, 0.5, 0)), (y == h // 2) & (x == w // 6)


def fixture(job, h, w=WIDTH):
    job.mkdir(parents=True, exist_ok=False)
    grid = job / "material.grid"
    grid.write_bytes(f"CFDKCYL1 {w} {h} 17\n0 0 0\n0 0 1\n1 0 0\n1 .5 .2\n".encode())
    table = np.zeros((3, 4), dtype="i8")
    missed = 0
    with ExitStack() as stack:
        files = {
            s: stack.enter_context((job / s).open("xb"))
            for s in [*(f"{r}-{i:02}.wall" for r in ROLES for i in range(3)), *MAPS]
        }
        for y in range(0, h, 16):
            stop = min(h, y + 16)
            d, aperture = field(range(y, stop), h, w)
            slices = [[], []]
            for role, r in enumerate(ROLES):
                for i in range(3):
                    t = np.full(d.shape, 3 + (i - 1) / 16, dtype="f8")
                    if role:
                        t -= d
                    if role and i == 2:
                        t[aperture] = np.nan
                    rec = np.zeros(d.shape, WALL_DTYPE)
                    rec["inner"] = np.where(np.isnan(t), np.nan, 2)
                    rec["outer"] = t + 2
                    rec["thickness"] = t
                    rec["status"] = np.isnan(t)
                    rec.tofile(files[f"{r}-{i:02}.wall"])
                    slices[role].append(t)
            ref, part = slices
            allpairs = np.stack([a - b for a, b in product(ref, part)])
            low, high = allpairs.min(0), allpairs.max(0)
            cent = allpairs[4] >= GATE
            code = np.select(
                [np.isnan(low), low >= GATE, high >= GATE], [3, 2, 1], default=0
            ).astype("u1")
            diag = allpairs[[0, 4, 8]] >= GATE
            missed += int(((code == 1) & (diag.all(0) == diag.any(0))).sum())
            table += np.bincount((4 * cent + code).ravel(), minlength=12).reshape(3, 4)
            for s, arr in zip(MAPS, (code, low, high), strict=True):
                arr.tofile(files[s])
    doc = dict(
        format="CFDK-synthetic-stripe-fixture-v1",
        scope="Algorithm stress only. Generated wall records, not a witnessed CT measurement.",
        shape_ha=[h, w],
        levels_per_mm=LEVELS,
        nominal_level_per_mm=LEVELS[1],
        class_codes=list(CLASSES),
        loss_gate_mm=GATE,
        independent_pairs=9,
        counts=dict(zip(CLASSES, table.sum(0).tolist(), strict=True)),
        nominal_by_class={
            n: dict(zip(CLASSES, row.tolist(), strict=True))
            for n, row in zip(("below", "flagged", "unknown"), table, strict=True)
        },
        sensitive_missed_by_same_level=missed,
        outputs_sha256={p.name: sha(p) for p in sorted(job.iterdir())},
    )
    write_doc(job / "manifest.json", doc)


def peak_working_set():
    if os.name == "nt":
        return int(
            subprocess.check_output(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    f"(Get-Process -Id {os.getpid()}).PeakWorkingSet64",
                ],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        )
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
        1 if sys.platform == "darwin" else 1024
    )


def worker(job, band, out):
    doc = json.loads((job / "manifest.json").read_bytes())
    require(doc["format"] == "CFDK-synthetic-stripe-fixture-v1", "synthetic harness only")
    h, w = doc["shape_ha"]
    summary, stats = scan(job, doc, sha(job / "manifest.json"), [[10, 0], [h - 11, w // 3]], band)
    rss = peak_working_set()
    write_doc(out, dict(result=summary, work=stats, os_peak_working_set_bytes=rss))


def dense(h, w):
    # Independent full-frame topology, no saved wall record or frontier. Fixed-point dyadic inputs.
    d, blind = field(range(h), h, w)
    pts = [(10, 0), (h - 11, w // 3)]
    records = []
    for i, j in product(range(3), repeat=2):
        unknown = blind if j == 2 else np.zeros_like(blind)
        hit = (d + (i - j) / 16 >= GATE) & ~unknown
        lab = labels(hit)
        opt = labels(hit | unknown)
        a, b = (lab[p] for p in pts)
        u, v = (opt[p] for p in pts)
        unique, n = np.unique(lab[lab > 0], return_counts=True)
        require(a and b, "stress anchors not measured")
        state = (
            "observed_link"
            if a == b
            else "unresolved_blind_path"
            if u == v
            else "separated_on_grid"
        )
        records.append(
            dict(
                pair=[i, j],
                observed_components=len(unique),
                largest_observed_cells=int(n.max()),
                observed_flagged_cells=int(hit.sum()),
                unknown_cells=int(unknown.sum()),
                anchor_cells=["flagged", "flagged"],
                anchor_state=state,
            )
        )
    return records


def qualify(root):
    root.mkdir(parents=True, exist_ok=False)
    cases = {}
    for h in (384, 1536):
        job = root / f"{h}x{WIDTH}"
        fixture(job, h)
        truth = dense(h, WIDTH)
        rows = []
        for band in (8, 32):
            out = root / f"worker-{h}-{band}.json"
            subprocess.run(
                [sys.executable, __file__, "worker", str(job), str(band), str(out)], check=True
            )
            doc = json.loads(out.read_bytes())
            require(doc["result"]["scenarios"] == truth, "full-frame oracle differs")
            stats = doc["work"]
            require(
                stats["peak_band_cells"] == WIDTH * band
                and stats["payload_bytes_read"] == 209 * h * WIDTH
                and stats["hashed_payloads"] == 9
                and stats["max_retained_components_per_frontier"] <= WIDTH
                and doc["os_peak_working_set_bytes"] < 256 << 20,
                "streaming work or memory gate",
            )
            rows.append(dict(**stats, os_peak_working_set_bytes=doc["os_peak_working_set_bytes"]))
            print(h, band, rows[-1]["os_peak_working_set_bytes"], flush=True)
        require(
            len({r["anchor_state"] for r in truth}) == 3,
            "stress topology should include measured link, blind bridge and separation",
        )
        cases[str(h)] = dict(
            synthetic_manifest_sha256=sha(job / "manifest.json"), dense_oracle=truth, trials=rows
        )
    require(
        WIDTH * 1536 > MAX_CELLS
        and all(
            b["os_peak_working_set_bytes"] - a["os_peak_working_set_bytes"] < 48 << 20
            for a, b in zip(cases["384"]["trials"], cases["1536"]["trials"], strict=True)
        ),
        "large-grid or fixed-band memory growth gate",
    )
    write_doc(
        root / "qualification.json",
        dict(
            scope="Synthetic walls only; input byte accounting excludes small manifest and grid. Each trial in a fresh process, OS peak working set includes Python, NumPy and SciPy. No full-frame labels or path returned by streaming worker.",
            width=WIDTH,
            full_detail_dense_limit_cells=MAX_CELLS,
            process_memory_gates=dict(max_peak_mib=256, max_growth_same_band_4x_height_mib=48),
            source_sha256={
                n: sha(Path(__file__).with_name(n))
                for n in ("stripe_review.py", "stripe_frontier.py", "qualify_stripe_scale.py")
            },
            cases=cases,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    q = sub.add_parser("qualify")
    q.add_argument("root", type=Path)
    q = sub.add_parser("worker")
    q.add_argument("job", type=Path)
    q.add_argument("band", type=int)
    q.add_argument("out", type=Path)
    a = p.parse_args()
    if a.mode == "qualify":
        qualify(a.root)
    else:
        worker(a.job, a.band, a.out)
