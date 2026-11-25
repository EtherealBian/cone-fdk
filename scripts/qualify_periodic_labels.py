"""Qualify a separate CUDA label backend against a frozen protocol and externally pinned data."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from itertools import product
from pathlib import Path
from statistics import median

import numpy as np
from build_periodic_labels import ROOT, sha
from common import dll_dirs
from qualify_anchors import unpack
from qualify_stripe_scale import GATE, field

PROTOCOL = "da12c4ed04dc6b05ba5b8904bb43fa6cfb6d29a61dab5972dbf323ba95a9b984"
REFERENCES = {
    "bridge-result.json": "d94b1e1a91e4c650a10e4907fa368556186ac0e740611d7c1b7aa6bee82d6962",
    "region-qualification.json": "f6c6c5d129b63ee1573eac34dcfb6429dd5305086580c1c1ebac5d0ace5f4b2d",
}


def need(ok, why):
    if not ok:
        raise ValueError(why)


def checked(p, digest):
    raw = p.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == digest, "source SHA-256: " + p.name)
    return json.loads(raw)


def flood(a):
    h, w = a.shape
    out = np.where(a == 2, -1, 0).astype("<i4")
    for y, x in np.ndindex(a.shape):
        if a[y, x] != 1 or out[y, x]:
            continue
        key = y * w + x + 1
        q = deque([(y, x)])
        out[y, x] = key
        while q:
            r, c = q.popleft()
            for rr, cc in ((r - 1, c), (r + 1, c), (r, (c + 1) % w), (r, (c - 1) % w)):
                if 0 <= rr < h and a[rr, cc] == 1 and not out[rr, cc]:
                    q.append((rr, cc))
                    out[rr, cc] = key
    return out


def masks_from_native(reports, bridge, legacy):
    records = {n: checked(reports / n, pin) for n, pin in REFERENCES.items()}
    for case in [
        "open-control",
        "critical-a",
        "critical-b",
        "joined-control",
        "invalid-edge-control",
    ]:
        old = case == "invalid-edge-control"
        row = records["region-qualification.json" if old else "bridge-result.json"]["cases"][case]
        seen = set()
        for r in row["cases" if old else "rows"]:
            i, j = r["pair"]
            need((i, j) not in seen, "duplicate pair")
            seen.add((i, j))
            path = (legacy if old else bridge) / case / f"native-{i}-{j}.json"
            lab = np.asarray(
                unpack(path, r["native_report_sha256" if old else "native_sha256"]), dtype="<i4"
            )
            need(
                hashlib.sha256(lab.tobytes()).hexdigest() == r["labels_i32le_sha256"],
                "native label digest",
            )
            yield np.where(lab > 0, 1, np.where(lab < 0, 2, 0)).astype("u1"), lab
        need(seen == set(product(range(2 if old else 3), repeat=2)), "source pairs")


def adversarial(h, w):
    # Not part of the timing workload. Multi-block dense contention and four-boundary counterexamples.
    y, x = np.indices((h, w))
    serpent = np.where(y % 2 == 0, 1, (x == 1) & (y % 4 == 1) | (x == w - 2) & (y % 4 == 3))
    return [
        np.ones((h, w), "u1"),
        serpent.astype("u1"),
        np.asarray((x + y) % 2 == 0, dtype="u1"),
        np.where((y == 0) | (y == h - 1), 1, 2).astype("u1"),
        np.zeros((h, w), "u1"),
    ]


def stress(h, w):
    d, blind = field(range(h), h, w)
    for i, j in product(range(3), repeat=2):
        a = np.asarray(d + (i - j) / 16 >= GATE, dtype="u1")
        if j == 2:
            a[blind] = 2
        yield a


def runproc(cmd, env, success=True):
    t = time.perf_counter()
    r = subprocess.run(
        list(map(str, cmd)),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf8",
        errors="replace",
        timeout=180,
    )
    seconds = time.perf_counter() - t
    if not success:
        need(r.returncode == 2, "expected a native refusal: " + r.stderr)
        return r.stderr.strip()
    if r.returncode:
        raise RuntimeError(r.stderr)
    return json.loads(r.stdout), seconds * 1000


def account(d, h, w, k, budget):
    n = h * w
    need(
        d["format"] == "CFDK-periodic-cuda-labels-v1"
        and d["maps"] == k
        and d["shape_ha"] == [h, w]
        and d["h2d_payload_bytes"] == k * n
        and d["d2h_payload_bytes"] == 4 * k * n
        and d["kernel_launches"] == 3 * k
        and d["redzones_verified"] is True
        and d["budget_bytes"] == budget << 20
        and d["peak_owned_bytes"] == (n + 3) // 4 * 4 + 8 * n + 1536
        and d["peak_owned_bytes"] <= d["budget_bytes"]
        and math.isfinite(d["gpu_label_event_ms"])
        and d["gpu_label_event_ms"] >= 0,
        "guarded device accounting",
    )


def performance(rows):
    need(
        len(rows) == 6
        and all(
            r["order"] == (["gpu", "cpu"] if i % 2 == 0 else ["cpu", "gpu"])
            for i, r in enumerate(rows)
        ),
        "AB/BA order",
    )
    for r in rows:
        need(
            all(
                math.isfinite(r[k]) and r[k] > 0
                for k in (
                    "cpu_process_ms",
                    "gpu_process_ms",
                    "cpu_label_ms",
                    "gpu_label_event_ms",
                )
            ),
            "nonpositive timing",
        )
    wins = sum(r["gpu_process_ms"] < r["cpu_process_ms"] for r in rows)
    gpu = median(r["gpu_process_ms"] for r in rows)
    cpu = median(r["cpu_process_ms"] for r in rows)
    return dict(
        paired_wins=wins,
        median_gpu_process_ms=gpu,
        median_cpu_process_ms=cpu,
        median_ratio=gpu / cpu,
        median_cpu_label_ms=median(r["cpu_label_ms"] for r in rows),
        median_gpu_label_event_ms=median(r["gpu_label_event_ms"] for r in rows),
        faster_on_this_case=(wins >= 5 and gpu / cpu <= 0.95),
    )


def qualify(build, reports, bridge, legacy, work):
    checked(ROOT / "fixtures/cuda-periodic-label-protocol-v1.json", PROTOCOL)
    rec = json.loads((build / "build.json").read_bytes())
    need(
        rec["identical_builds"] == 2
        and set(rec["products"]) == {"periodic-labels.exe", "periodic-labels.ptx"}
        and all(sha(build / n) == v for n, v in rec["products"].items()),
        "repeat build products",
    )
    sources = {
        n: sha(ROOT / ("scripts" if n.endswith(".py") else "src") / n)
        for n in [
            "periodic_labels.cpp",
            "periodic_labels.cu",
            "files.hpp",
            "gpu.hpp",
            "build_periodic_labels.py",
        ]
    }
    need(sources == rec["sources"], "repeat build sources")
    work.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PATH=os.pathsep.join(map(str, dll_dirs())) + os.pathsep + os.environ.get("PATH", ""),
    )

    def gcmd(src, dst, h, w, k, b=32):
        return [
            build / "periodic-labels.exe",
            "--input",
            src,
            "--output",
            dst,
            "--ptx",
            build / "periodic-labels.ptx",
            "--height",
            h,
            "--width",
            w,
            "--maps",
            k,
            "--budget",
            b,
        ]

    data = {}

    def known(name, items):
        src = work / (name + ".u8")
        truth = work / (name + "-expected.i32")
        out = work / (name + ".i32")
        shape = None
        k = 0
        with src.open("xb") as f, truth.open("xb") as g:
            for a, lab in items:
                need(
                    a.shape == lab.shape and (shape is None or a.shape == shape),
                    "different shapes",
                )
                shape = a.shape
                f.write(a.tobytes())
                g.write(lab.astype("<i4").tobytes())
                k += 1
        h, w = shape
        d, _ = runproc(gcmd(src, out, h, w, k), env)
        account(d, h, w, k, 32)
        need(sha(out) == sha(truth), name + " output differs")
        data[name] = dict(
            shape_ha=[h, w],
            maps=k,
            pair_cells=k * h * w,
            input_sha256=sha(src),
            output_sha256=sha(out),
            peak_owned_bytes=d["peak_owned_bytes"],
            redzones_verified=True,
        )

    small = (np.asarray(t, dtype="u1").reshape(2, 3) for t in product(range(3), repeat=6))
    known("tiny", ((a, flood(a)) for a in small))
    known("native", masks_from_native(reports, bridge, legacy))
    known("contention", ((a, flood(a)) for a in adversarial(257, 263)))
    known("single-row", ((a, flood(a)) for a in adversarial(1, 4096)))
    need(
        data["tiny"]["maps"] == 729
        and data["native"]["maps"] == 40
        and data["native"]["pair_cells"] == 590400,
        "map scope",
    )
    h, w, k = 1536, 960, 9
    src = work / "scale.u8"
    out = work / "scale.i32"
    g = gcmd(src, out, h, w, k)
    c = [sys.executable, ROOT / "scripts/periodic_label_cpu.py", src, out, h, w, k]
    with src.open("xb") as f:
        for a in stress(h, w):
            f.write(a.tobytes())

    # CPU truth and three fresh GPU runs are outside the six measured pairs.
    def single(mode, pin):
        d, ms = runproc(g if mode == "gpu" else c, env)
        if mode == "gpu":
            account(d, h, w, k, 32)
        need(out.stat().st_size == 4 * h * w * k, "output size")
        digest = sha(out)
        need(pin is None or digest == pin, "large output differs")
        out.unlink()
        return d, ms, digest

    _, _, pin = single("cpu", None)
    repeated = []
    for _ in range(3):
        device, _, sig = single("gpu", pin)
        repeated.append(sig)
    rows = []
    for i in range(6):
        order = ["gpu", "cpu"] if i % 2 == 0 else ["cpu", "gpu"]
        row = dict(order=order)
        for mode in order:
            d, ms, _ = single(mode, pin)
            row[mode + "_process_ms"] = ms
            key = "cpu_label_ms" if mode == "cpu" else "gpu_label_event_ms"
            row[key] = d[key]
        rows.append(row)
    refuse = {}

    def no(name, cmd, message, preserve=False):
        if preserve:
            out.write_bytes(b"previous valid destination")
        text = runproc(cmd, env, False)
        need(message in text, "wrong refusal: " + text)
        need(
            (out.read_bytes() == b"previous valid destination" if preserve else not out.exists())
            and not list(work.glob(out.name + ".partial.*")),
            "refusal leaked or damaged output",
        )
        refuse[name] = text
        if preserve:
            out.unlink()

    no("existing", g, "destination already exists", True)
    no("budget", gcmd(src, out, h, w, k, 1), "budget cannot fit")
    broken = work / "corrupt.u8"
    broken.write_bytes(b"\x01" * 11)
    cmd = gcmd(broken, out, 2, 3, 2)
    no("truncated", cmd, "size mismatch")
    broken.write_bytes(b"\x01" * 11 + b"\xff")
    no("invalid_last_map", cmd, "invalid ternary byte in scenario 1")
    data["scale"] = dict(
        shape_ha=[h, w],
        maps=k,
        pair_cells=h * w * k,
        input_sha256=sha(src),
        output_sha256=pin,
        repeated_gpu_output_sha256=repeated,
        peak_owned_bytes=device["peak_owned_bytes"],
        redzones_verified=True,
    )
    final = dict(
        format="CFDK-periodic-label-qualification-v1",
        protocol_sha256=PROTOCOL,
        build=rec,
        source_references=REFERENCES,
        gpu=device["gpu"],
        driver=device["driver"],
        workloads=data,
        refusals=refuse,
        verification_sources={
            n: sha(ROOT / "scripts" / n)
            for n in (
                "periodic_label_cpu.py",
                "qualify_periodic_labels.py",
                "qualify_stripe_scale.py",
                "qualify_anchors.py",
                "region_review.py",
            )
        },
        timing_boundary="Fresh process creation through exit, including runtime imports or driver initialization, preclassified mask reads, durable dense file flush and publication. CPU uses the repository SciPy/NumPy implementation. Six AB/BA pairs; warm file cache, unlocked clocks, output SHA outside timer. Secondary timers exclude file I/O and startup. No detector, wall classification or metrology timing.",
        pairs=rows,
        timing=performance(rows),
    )
    (work / "qualification.json").write_bytes((json.dumps(final, indent=2) + "\n").encode())
    print(json.dumps(final["timing"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("build", "reports", "bridge", "legacy", "work"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
