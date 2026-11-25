"""Qualify the bounded-halo CUDA interface encoder against discrete source neighbors."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
from anchor_audit import publish
from build_periodic_labels import ROOT, sha
from common import dll_dirs
from interface_ledger import codes
from qualify_periodic_labels import need, runproc

PROTOCOL = "72ba73a2a80783466ac3fad9c0770455ef3e2d373471de5ddcf17b70c89cb2f1"
SCALE = "31a890fbfa3aadbb134a4876cb72eed2854b4557447525d205b679110f30693a"
BUILD_NAMES = {"periodic-interfaces.exe", "periodic-interfaces.ptx"}


def direct(a):
    """Pointwise, ordinary integer oracle with no rolling or halo machinery."""
    h, w = a.shape
    out = np.zeros((h, w), np.uint8)
    table = {0: 1, 1: 0, 2: 2}
    for y, x in np.ndindex(h, w):
        if a[y, x] != 1:
            continue
        for d, (r, c) in enumerate(((y - 1, x), (y, (x + 1) % w), (y + 1, x), (y, (x - 1) % w))):
            val = 3 if r < 0 or r >= h else table[int(a[r, c])]
            out[y, x] |= np.uint8(val << (2 * d))
    return out


def vector(a):
    return codes(np.pad(a, ((1, 1), (0, 0)), constant_values=3))


def native(build, src, dest, h, w, maps, band, budget=1, *, refusal=None):
    cmd = [
        build / "periodic-interfaces.exe",
        "--ptx",
        build / "periodic-interfaces.ptx",
        "--input",
        src,
        "--output",
        dest,
        "--height",
        h,
        "--width",
        w,
        "--maps",
        maps,
        "--band",
        band,
        "--budget",
        budget,
    ]
    env = dict(
        os.environ,
        PATH=os.pathsep.join(map(str, dll_dirs())) + os.pathsep + os.environ.get("PATH", ""),
    )
    if refusal:
        msg = runproc(cmd, env, False)
        need(refusal in msg, "wrong native refusal: " + msg)
        return msg
    d, _ = runproc(cmd, env)
    account(d, h, w, maps, band, budget)
    need(dest.stat().st_size == maps * h * w, "native interface file length")
    return d


def account(d, h, w, maps, band, budget):
    need(
        type(band) is int and 1 <= band <= 4096 and type(budget) is int and 1 <= budget <= 65536,
        "stored CUDA band or budget",
    )
    launches = maps * ((h + band - 1) // band)
    size = maps * h * w
    cap = min(h, band)
    owned = 4 * ((w * (cap + 2) + 3) // 4 + (w * cap + 3) // 4) + 1024
    need(
        d["format"] == "CFDK-periodic-interfaces-v1"
        and d["shape_ha"] == [h, w]
        and d["maps"] == maps
        and d["band_rows"] == band
        and d["read_bytes"] == d["d2h_bytes"] == size
        and d["kernel_launches"] == launches
        and d["h2d_bytes"] == size + 2 * w * launches
        and d["peak_owned_bytes"] == owned <= d["budget_bytes"] == budget << 20
        and d["redzones_verified"] is True,
        "native interface accounting",
    )


def qualify(build, scale, work):
    build, scale, work = (Path(x).resolve() for x in (build, scale, work))
    need(
        sha(ROOT / "fixtures/periodic-interface-protocol-v1.json") == PROTOCOL
        and sha(scale) == SCALE,
        "different precommitted rules or pressure masks",
    )
    rec = json.loads((build / "build.json").read_bytes())
    need(
        rec["format"] == "CFDK-periodic-interface-build-v1"
        and rec["identical_builds"] == 2
        and set(rec["products"]) == BUILD_NAMES
        and all(sha(build / n) == v for n, v in rec["products"].items()),
        "different interface build",
    )
    work.mkdir(parents=True, exist_ok=False)
    workloads = {}
    devices = set()

    def trial(name, src, k, h, w, expected):
        runs = []
        for band in (1, 7, 32):
            dst = work / f"{name}-b{band}.u8"
            d = native(build, src, dst, h, w, k, band)
            devices.add((d.pop("gpu"), d.pop("driver")))
            need(sha(dst) == expected, "different interface codes: " + name)
            runs.append(d)
        workloads[name] = dict(
            shape_ha=[h, w],
            maps=k,
            cells=k * h * w,
            input_sha256=sha(src),
            face_sha256=expected,
            runs=runs,
        )

    def sample(name, arrays):
        src = work / (name + ".u8")
        a = np.asarray(arrays, dtype="u1")
        a.tofile(src)
        truth = hashlib.sha256()
        for one in a:
            oracle = direct(one)
            need(np.array_equal(oracle, vector(one)), "independent CPU oracles disagree")
            truth.update(oracle.tobytes())
        trial(name, src, *a.shape, truth.hexdigest())

    sample("tiny", [np.asarray(t).reshape(2, 3) for t in itertools.product(range(3), repeat=6)])
    sample("one-row", [[[1, 0, 1, 1, 2, 2, 1]], [[1] * 7], [[0] * 7], [[2] * 7]])
    ring = np.zeros((8, 17), "u1")
    ring[3] = 1
    hole = np.ones_like(ring)
    hole[4, 8] = 0
    blind = hole.copy()
    blind[4, 8] = 2
    ends = np.zeros_like(ring)
    ends[[0, -1], 0] = 1
    seam = np.zeros_like(ring)
    seam[2, [0, -1]] = 1
    sample("analytic", [ring, hole, blind, ends, seam])
    # Same width with two distinct heights, both larger than every requested band.
    for h in (47, 211):
        y, x = np.indices((h, 83))
        sample(f"height-{h}", [(x + y) % 3, (3 * y + x // 3) % 3])
    need(
        [r["peak_owned_bytes"] for r in workloads["height-47"]["runs"]]
        == [r["peak_owned_bytes"] for r in workloads["height-211"]["runs"]],
        "height changed owned device peak",
    )
    h, w, k = 1536, 960, 9
    truth = hashlib.sha256()
    with scale.open("rb") as f:
        for _ in range(k):
            b = f.read(h * w)
            need(len(b) == h * w, "short pressure mask")
            truth.update(vector(np.frombuffer(b, "u1").reshape(h, w)).tobytes())
        need(not f.read(1), "trailing pressure mask")
    trial("scale", scale, k, h, w, truth.hexdigest())
    need(len(devices) == 1, "mixed GPUs")
    bad = work / "bad.u8"
    out = work / "refused.u8"
    fails = {}
    for why, shape, data, text in [
        ("last", (2, 3, 2, 1), b"\1" * 11 + b"\3", "invalid ternary byte"),
        ("short", (2, 3, 2, 7), b"\0" * 11, "input size mismatch"),
        ("budget", (256, 4096, 1, 256), b"", "device budget cannot fit"),
    ]:
        bad.write_bytes(data)
        fails[why] = native(build, bad, out, *shape, refusal=text)
        need(
            not out.exists() and not list(work.glob("refused.u8.partial.*")),
            "refusal left a result",
        )
    bad.write_bytes(b"\1" * 6)
    out.write_bytes(b"preserve")
    fails["existing"] = native(build, bad, out, 2, 3, 1, 1, refusal="destination already exists")
    need(out.read_bytes() == b"preserve", "changed prior result")
    fails["same_path"] = native(build, bad, bad, 2, 3, 1, 1, refusal="destination already exists")
    need(bad.read_bytes() == b"\1" * 6, "changed source")
    gpu, driver = devices.pop()
    publish(
        work / "qualification.json",
        dict(
            format="CFDK-periodic-interface-qualification-v1",
            scope="Typed sampled-cell interfaces only. Memory accounting includes owned guarded CUDA buffers, not the runtime, file cache, or host. No speed or physical accuracy claim.",
            protocol_sha256=PROTOCOL,
            build=rec,
            workloads=workloads,
            gpu=gpu,
            driver=driver,
            refusals=fails,
            sources={
                n: sha(ROOT / "scripts" / n)
                for n in (
                    "qualify_periodic_interfaces.py",
                    "interface_ledger.py",
                    "qualify_periodic_labels.py",
                )
            },
        ),
    )
    print("PASS: all three band heights, pressure maps, typed edges and native refusal controls")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("build", "scale", "work"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
