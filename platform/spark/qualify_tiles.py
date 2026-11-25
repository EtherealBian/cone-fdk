"""Exercise tails, resource limits and driver-loaded GB10 backprojection specializations."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import spark_fdk as s
from spark_fdk import t, need, M
from spark_publication import fresh, publish, staging

BOXES = {
    "dense": None,
    "roi": [1, 2, 3, 29, 31, 23],
    "corner": [62, 60, 59, 1, 3, 4],
    "multipass": None,
}


def inputs(wd):
    g, p = wd / "synthetic.geom", wd / "synthetic.f32"
    g.write_text("CFDK1 13 33 37 0.7 0.8 100 150 0.273 -1\n")
    i = np.arange(13 * 33 * 37, dtype=np.int64)
    values = ((i * 137 % 1009) - 504 + (i % 29) * 2).astype("<f4") / np.float32(1024)
    values.tofile(p)
    return g, p


def command(tile, case, g, p, o):
    native = tile == "original"
    b = t.ROOT / "build/spark" if native else M / "tile-fdk-build-001"
    box = BOXES[case]
    cmd = [
        b / ("cone-fdk" if native else "tile-fdk"),
        "--ptx",
        b / ("kernels.ptx" if native else "tile.ptx"),
        "--geom",
        g,
        "--input",
        p,
        "--output",
        o,
        "--n",
        "63",
        "--fov",
        "18.4",
        "--budget",
        ("1" if case == "multipass" else "4"),
        "--batch",
        "8",
    ]
    if box is not None:
        cmd += ["--roi", ",".join(map(str, box))]
    if not native:
        cmd += ["--gb10-tile", tile]
    return list(map(str, cmd))


def native_check(tile, case, stdout):
    a = json.loads(stdout)
    box = BOXES[case] or [0, 0, 0, 63, 63, 63]
    need(
        a["redzones_verified"] is True
        and a["gpu"] == "NVIDIA GB10"
        and a["views"] == 13
        and a["batch"] == 8
        and a["filter_batches"] == 2 * a["projection_passes"]
        and a["projection_passes"] == a["slabs"]
        and ((a["slabs"] > 1) if case == "multipass" else a["slabs"] == 1)
        and a["output_shape_xyz"] == box[3:]
        and a["roi_origin_xyz"] == box[:3]
        and a["output_bytes"] == int(np.prod(box[3:])) * 4,
        "wrong analytic workload",
    )
    need(
        a.get("gb10_tile") == (None if tile == "original" else list(map(int, tile.split("x")))),
        "wrong native tile",
    )
    return a


def qualification(out):
    out = fresh(out)
    p, b, plan, n, a = s.evidence()
    best, _ = s.attest(p, b, plan, n, a)
    bp = M / "tile-fdk-build-001"
    need(
        all(t.sha(bp / k) == b["independent_products"][0][k] for k in ["tile.ptx", "tile-fdk"])
        and t.sha(t.ROOT / "build/spark/cone-fdk") == t.BASE
        and t.sha(t.ROOT / "build/spark/kernels.ptx") == t.PTX,
        "build changed",
    )
    tool = "/usr/local/cuda/bin/compute-sanitizer"
    rows = []
    san = []
    with staging(out, ".gb10-tile-guards-") as tmp:
        stage = Path(tmp) / "ready"
        stage.mkdir()
        with tempfile.TemporaryDirectory(dir=Path(tmp).resolve(strict=True)) as temp:
            wd = Path(temp)
            g, pr = inputs(wd)
            source = t.HERE / "fdk/tile_attributes.cpp"
            probe = wd / "probe"
            subprocess.run(
                [
                    "g++",
                    "-O2",
                    "-std=c++17",
                    "-I/usr/local/cuda/include",
                    source,
                    "-L/usr/local/cuda/lib64/stubs",
                    "-lcuda",
                    "-o",
                    probe,
                ],
                check=True,
            )
            attr = [
                json.loads(line)
                for line in subprocess.check_output(
                    [probe, bp / "tile.ptx"], text=True
                ).splitlines()
            ]
            need(
                len(attr) == 18
                and all(
                    r["binary_version"] == 121
                    and 0 < r["threads"] * r["active_blocks_bound"] <= r["max_threads_per_sm"]
                    and r["registers"] > 0
                    for r in attr
                ),
                "bad resource counters",
            )
            for c, box in BOXES.items():
                ref = wd / (c + ".f32")
                pp = subprocess.run(
                    command("original", c, g, pr, ref), capture_output=True, timeout=40
                )
                need(not pp.returncode and not pp.stderr, pp.stderr.decode(errors="replace"))
                nr = native_check("original", c, pp.stdout)
                h = t.sha(ref)
                data = np.fromfile(ref, "<f4")
                need(
                    np.isfinite(data).all()
                    and (data.any() if c == "corner" else data.min() < 0 < data.max()),
                    "degenerate fixture",
                )
                if box:
                    x, y, z, nx, ny, nz = box
                    crop = np.memmap(wd / "dense.f32", mode="r", dtype="<f4", shape=(63,) * 3)[
                        z : z + nz, y : y + ny, x : x + nx
                    ]
                    need(crop.tobytes() == data.tobytes(), "frozen crop mismatch")
                for tile in map(t.tile_name, t.TILES):
                    fn = wd / "candidate"
                    pp = subprocess.run(
                        command(tile, c, g, pr, fn), capture_output=True, timeout=40
                    )
                    need(not pp.returncode and not pp.stderr, pp.stderr.decode(errors="replace"))
                    r = native_check(tile, c, pp.stdout)
                    v = np.fromfile(fn, "<f4")
                    need(
                        v.shape == data.shape
                        and np.isfinite(v).all()
                        and r["peak_owned_bytes"] == nr["peak_owned_bytes"],
                        "invalid synthetic result",
                    )
                    mask = v.view("<u4") != data.view("<u4")
                    idx = np.flatnonzero(mask)
                    h2 = t.sha(fn)
                    need((len(idx) == 0) == (h == h2), "hash/bits disagreement")
                    first = int(idx[0]) if len(idx) else None
                    rows.append(
                        dict(
                            case=c,
                            tile=tile,
                            reference_sha256=h,
                            sha256=h2,
                            slabs=r["slabs"],
                            bytes=r["output_bytes"],
                            different_bits_voxels=len(idx),
                            max_abs=float(np.max(np.abs(v.astype("f8") - data))),
                            first_flat=first,
                            first_u32=None
                            if first is None
                            else [int(data.view("<u4")[first]), int(v.view("<u4")[first])],
                        )
                    )
                    if tile == best:
                        need(not len(idx), "profiled tile changed output bits")
                    fn.unlink()
                if c not in ("dense", "roi"):
                    continue
                for which in ("memcheck", "synccheck"):
                    fn = wd / "sanitized"
                    log = stage.resolve(strict=True) / f"{c}-{which}.log"
                    cmd = [
                        tool,
                        "--tool",
                        which,
                        "--log-file",
                        str(log),
                        "--error-exitcode",
                        "79",
                    ]
                    pp = subprocess.run(
                        cmd + command(best, c, g, pr, fn),
                        capture_output=True,
                        timeout=180,
                        env=dict(os.environ, LC_ALL="C"),
                    )
                    text = log.read_text() if log.is_file() else ""
                    need(
                        pp.returncode == 0
                        and not pp.stderr
                        and "ERROR SUMMARY: 0 errors" in text,
                        which + ": " + pp.stderr.decode(errors="replace")[-300:] + text[-500:],
                    )
                    native_check(best, c, pp.stdout)
                    need(t.sha(fn) == h, "sanitizer output mismatch")
                    fn.unlink()
                    san.append(
                        dict(
                            case=c, tool=which, tile=best, output_sha256=h, log_sha256=t.sha(log)
                        )
                    )
            doc = dict(
                format="CFDK-GB10-tile-qualification-v1",
                profile=s.PROFILE,
                build=t.BUILD,
                script_sha256=t.sha(__file__),
                fixture_sha256=[t.sha(g), t.sha(pr)],
                probe_source_sha256=t.sha(source),
                resource_attributes=attr,
                comparisons=rows,
                sanitizers=san,
                eligible_by_bits=[
                    name
                    for name in map(t.tile_name, t.TILES)
                    if all(r["different_bits_voxels"] == 0 for r in rows if r["tile"] == name)
                ],
                scope="Analytic signed inputs, 13 views in two batches, 63 cubed and odd cropped/tight-budget outputs. Multipass is compared to the original with the same budget; no cross-budget bitwise assumption. Driver occupancy is a theoretical resource limit, not observed occupancy or attribution of a speedup.",
            )
            (stage / "qualification.json").write_text(json.dumps(doc, indent=2) + "\n")
        publish(stage, out)
    print(t.sha(out / "qualification.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    qualification(parser.parse_args().out)
