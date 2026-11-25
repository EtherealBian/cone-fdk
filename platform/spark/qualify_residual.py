"""Compare every plane on CUDA and independently on CPU; publish only after the source gates."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from build_residual import HERE, ROOT, RULES, sha
from residual_witness import CUTS, FIELDS, agree, bounded, hashed, need, pin
from sealed_inputs import snapshot
from spark_publication import fresh, publish, staging

BUILD = "b83e045b2e66612a48bde9bd92a0864ad26fc0bde08e5417b029478f0da6cd3c"
ASTRA = "fa5180be58a97e86e4be69c7515a6e8937b98c0a57ba2a3e6b402030a8aa4df4"
NATIVE = "c1320694248460d2de39f892f377b1b3a7473d186f32d9971d789d45d140f0ec"
PINE = "4fd5f98f528e467532cefa8c498786e6e28fd662b2e09b4a92a8db77b4d69218"
PINE_SOURCE = "0e236a75a8d4ebe9dc09b1f1a79f8ab832c42df8738340d66b40ba13d4c5deb4"


def json_checked(path, expected):
    return json.loads(bounded(path, 2 << 20, expected))


def volumes(kind, m, pine_pin=None):
    if kind == "walnut720":
        need(pine_pin is None, "unexpected pine qualification")
        aroot = m / "walnut-astra-gb10-001"
        a = json_checked(aroot / "result.json", ASTRA)
        n = json_checked(m / "walnut-gb10-001/result.json", NATIVE)
        left = a["input_sha256"]["full-corrected-512.f32"]
        right = a["sha256"]
        need(
            a["passed"] is True
            and n["output_sha256"] == [left, left]
            and n["input_sha256"] == a["input_sha256"]
            and a["astra"]["astra_version"] == "2.5.0",
            "walnut provenance",
        )
        return dict(
            shape=[512] * 3,
            case=kind,
            parents={"native": NATIVE, "independent_astra": ASTRA},
            pins=[left, right],
            old=a["native_vs_spark_astra"],
            paths=[m / "reference/full-corrected-512.f32", aroot / "astra-gb10-512.f32"],
        )
    need(kind in ("pine90", "pine720"), "unknown physical case")
    pdir = m / "pine-gb10-001"
    q = json_checked(pdir / "qualification.json", pin(pine_pin))
    proto = json_checked(ROOT / "fixtures/pine-cone-qualification-protocol-v1.json", PINE)
    need(
        q["format"] == "CFDK-Spark-physical-pine-qualification-v1"
        and q["protocol_sha256"] == PINE
        and q["source_manifest_sha256"] == PINE_SOURCE
        and set(q["scans"]) == {"pine90", "pine720"}
        and q["script_sha256"] == sha(HERE / "qualify_pine.py"),
        "pine provenance",
    )
    row = q["scans"][kind]
    need(
        row["views"] == int(kind[4:])
        and row["dense_sha256"] == row["repeat_sha256"]
        and row["voxels"] == proto["reconstruction"]["n"] ** 3
        and len(row["rois"]) == 2,
        "pine dense gate",
    )
    return dict(
        shape=[proto["reconstruction"]["n"]] * 3,
        case=kind,
        parents={"physical_qualification": pine_pin, "protocol": PINE},
        pins=[row["dense_sha256"], row["astra_sha256"]],
        old=row["native_vs_astra"],
        paths=[pdir / kind / "native-0.f32", pdir / kind / "astra.f32"],
    )


def numpy_oracle(a, b):
    x = np.frombuffer(a, "<f4").astype("f8")
    y = np.frombuffer(b, "<f4").astype("f8")
    need(
        x.size and x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(),
        "invalid dense values",
    )
    d = x - y
    v = np.abs(d)
    return [
        x.size,
        int(np.count_nonzero(x == y)),
        *[int(np.count_nonzero(v > c)) for c in CUTS],
        float(v.max()),
        int(v.argmax()),
        float(d.sum()),
        float(v.sum()),
        float((d * d).sum()),
        float((y * y).sum()),
    ]


def accounts(native, shape, band, budget):
    nx, ny, nz = shape
    n = nx * ny
    tiles = (n + 1023) // 1024
    cap = min(nz, band)
    size = 72  # float64 x5, uint32 x8
    need(
        native["format"] == "CFDK-GB10-volume-residual-v1"
        and native["gpu"] == "NVIDIA GB10"
        and native["shape_xyz"] == shape
        and native["band"] == cap
        and native["tile_size"] == 1024
        and native["partial_bytes"] == size,
        "wrong kernel ABI or workload",
    )
    need(
        native["owned_bytes"] == cap * (8 * n + tiles * size) + 6 * 256 <= budget << 20
        and native["budget_bytes"] == budget << 20
        and native["host_payload_bytes"] == cap * (8 * n + tiles * size) + nz * size
        and native["guard_checks"] == 6
        and native["h2d_bytes"] == 8 * n * nz
        and native["d2h_bytes"] == tiles * nz * size
        and native["launches"] == (nz + cap - 1) // cap,
        "device accounting failed",
    )
    need(
        len(native["rows"]) == nz
        and all(
            type(native[k]) in (float, int) and math.isfinite(native[k]) and native[k] >= 0
            for k in ["kernel_ms", "read_ms", "wall_ms"]
        ),
        "bad rows or timing",
    )


def run_gpu(exe, fds, shape, band, budget):
    paths = [f"/proc/self/fd/{f}" for f in [exe, *fds]]
    start = time.monotonic_ns()
    result = subprocess.run(
        [*paths, *map(str, [*shape, band, budget])],
        pass_fds=(exe, *fds),
        capture_output=True,
        timeout=180,
    )
    elapsed = (time.monotonic_ns() - start) / 1e6
    need(
        result.returncode == 0,
        "native residual audit refused: " + result.stderr.decode(errors="replace")[:400],
    )
    r = json.loads(result.stdout)
    accounts(r, shape, band, budget)
    r["child_elapsed_ms"] = elapsed
    return r


def summarize(rows, nx, ny):
    n = sum(r[0] for r in rows)
    z = max(range(len(rows)), key=lambda j: rows[j][6])
    i = rows[z][7]
    sse = math.fsum(r[10] for r in rows)
    ref = math.fsum(r[11] for r in rows)
    return dict(
        voxels=n,
        numeric_equal=sum(r[1] for r in rows),
        strict_exceedance=[sum(r[i] for r in rows) for i in range(2, 6)],
        max_abs=rows[z][6],
        first_maximum_xyz=[i % nx, i // nx, z],
        signed_mean=math.fsum(r[8] for r in rows) / n,
        mean_abs=math.fsum(r[9] for r in rows) / n,
        rms=math.sqrt(sse / n),
        rel_l2=None if ref == 0 else math.sqrt(sse / ref),
    )


def qualify(build, job, m, case, pine_pin=None):
    job = fresh(job)
    spec = volumes(case, m, pine_pin)
    paths = spec.pop("paths")
    nx, ny, nz = shape = spec["shape"]
    size = 4 * nx * ny * nz
    for p in (*paths, build, m / "work"):
        need(
            not job.is_relative_to(p.resolve()) and not p.resolve().is_relative_to(job),
            "job overlaps a source",
        )
    br = json_checked(build / "build.json", BUILD)
    need(
        br["protocol_sha256"] == RULES
        and sha(ROOT / "fixtures/volume-residual-protocol-v1.json") == RULES
        and br["build_script_sha256"] == sha(HERE / "build_residual.py")
        and br["source_sha256"] == sha(HERE / "residuals/volume_residual.cu")
        and len(br["independent_elf_sha256"]) == 2
        and len(set(br["independent_elf_sha256"])) == 1,
        "native build source gate",
    )
    with staging(job, ".paired-residual-") as tmp, ExitStack() as stack:
        stage = Path(tmp) / "accepted"
        stage.mkdir()
        t = time.monotonic_ns()
        fds = [
            stack.enter_context(snapshot(p, h, limit=512 << 20, exact=size))
            for p, h in zip(paths, spec["pins"], strict=True)
        ]
        exe = stack.enter_context(
            snapshot(
                build / "volume-residual",
                br["independent_elf_sha256"][0],
                limit=4 << 20,
                executable=True,
            )
        )
        seal_ms = (time.monotonic_ns() - t) / 1e6
        native = run_gpu(exe, fds, shape, 7, 32)
        alt = run_gpu(exe, fds, shape, 1, 32)
        rows = native.pop("rows")
        need(rows == alt.pop("rows"), "different results across fresh processes or band sizes")
        s = summarize(rows, nx, ny)
        need(
            s["max_abs"] == spec["old"]["max_abs"]
            and s["rel_l2"] is not None
            and math.isclose(s["rel_l2"], spec["old"]["rel_l2"], rel_tol=2e-13, abs_tol=0),
            "frozen earlier global comparison disagrees",
        )
        planes = []
        worst = s["first_maximum_xyz"][2]
        t = time.monotonic_ns()
        for z, r in enumerate(rows):
            a, b = [os.pread(fd, nx * ny * 4, z * nx * ny * 4) for fd in fds]
            need(len(a) == len(b) == nx * ny * 4, "short source slice")
            cpu = numpy_oracle(a, b)
            agree(r, cpu)
            planes.append(
                dict(z=z, candidate_sha256=hashed(a), reference_sha256=hashed(b), gpu=r)
            )
            if z == worst:
                (stage / "candidate-slice.f32").write_bytes(a)
                (stage / "reference-slice.f32").write_bytes(b)
        oracle_ms = (time.monotonic_ns() - t) / 1e6
        doc = dict(
            format="CFDK-paired-volume-qualification-v1",
            protocol_sha256=RULES,
            build_sha256=BUILD,
            harness_sha256=sha(__file__),
            witness_tool_sha256=sha(HERE / "residual_witness.py"),
            source=spec,
            shape_xyz=shape,
            row_fields=FIELDS,
            slices_verified_by_numpy=nz,
            accounting=[native, alt],
            source_snapshot_ms=seal_ms,
            independent_cpu_scan_and_hash_ms=oracle_ms,
            aggregate=s,
            planes=planes,
            scope="Voxelwise same-grid numerical agreement; not measurement against an independently calibrated physical object.",
        )
        rpt = stage / "qualification.json"
        rpt.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        rp = sha(rpt)
        args = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(HERE / "residual_witness.py"),
            str(rpt),
            rp,
            str(worst),
            str(stage / "candidate-slice.f32"),
            str(stage / "reference-slice.f32"),
        ]
        proc = subprocess.run(args, check=True, capture_output=True, cwd=stage)
        w = json.loads(proc.stdout)
        need(
            w["maximum_xyz"] == s["first_maximum_xyz"] and w["max_abs"] == s["max_abs"],
            "offline worst-plane witness failed",
        )
        (stage / "witness.json").write_bytes(proc.stdout)
        publish(stage, job)
    return dict(report_sha256=rp, aggregate=s)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", choices=["walnut720", "pine90", "pine720"])
    p.add_argument("build", type=Path)
    p.add_argument("job", type=Path)
    p.add_argument("--pine-pin")
    a = p.parse_args()
    print(json.dumps(qualify(a.build, a.job, ROOT.parent, a.case, a.pine_pin)))
