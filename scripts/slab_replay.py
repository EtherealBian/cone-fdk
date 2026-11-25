"""Resume a baselined CUDA volume inside one specimen. Committed slabs live in SQLite BLOBs."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

HEX = re.compile(r"[0-9a-f]{64}")
STEP = 1 << 20
LIMIT = 256 << 20
NAMES = ("cone-fdk.exe", "kernels.ptx")
MARKER = b"CFDK private slab replay cache v1\n"


def need(test, why):
    if not test:
        raise ValueError(why)


def pin(value):
    need(type(value) is str and HEX.fullmatch(value), "external lowercase SHA-256 required")
    return value


def bounded(path, limit):
    with path.open("rb") as f:
        data = f.read(limit + 1)
    need(len(data) <= limit, "document too large")
    return data


def pairs(items):
    d = dict(items)
    need(len(d) == len(items), "duplicate JSON key")
    return d


def encode(d):
    return (json.dumps(d, indent=2, allow_nan=False) + "\n").encode("ascii")


def plan(path, expected):
    raw = bounded(path, STEP)
    need(hashlib.sha256(raw).hexdigest() == pin(expected), "plan SHA-256 mismatch")
    d = json.loads(raw, object_pairs_hook=pairs)
    need(
        isinstance(d, dict)
        and set(d)
        == {
            "format",
            "n",
            "fov_mm",
            "budget_mib",
            "batch",
            "geometry_sha256",
            "projections_sha256",
            "projections_bytes",
            "products",
            "volume_sha256",
            "slabs",
        }
        and d["format"] == "CFDK-slab-replay-plan-v1",
        "plan schema",
    )
    for k, lo, hi in (
        ("n", 1, 4096),
        ("batch", 1, 65536),
        ("budget_mib", 1, 65536),
        ("projections_bytes", 4, 1 << 43),
    ):
        need(type(d[k]) is int and lo <= d[k] <= hi, "invalid " + k)
    need(
        d["projections_bytes"] % 4 == 0
        and type(d["fov_mm"]) in (int, float)
        and 0 < d["fov_mm"] <= 1e6
        and math.isfinite(d["fov_mm"]),
        "projection length or fov",
    )
    need(type(d["products"]) is dict and set(d["products"]) == set(NAMES), "native product list")
    for value in (
        d["geometry_sha256"],
        d["projections_sha256"],
        d["volume_sha256"],
        *d["products"].values(),
    ):
        pin(value)
    n = d["n"]
    need(isinstance(d["slabs"], list) and 1 <= len(d["slabs"]) <= n, "slab count")
    z = 0
    for s in d["slabs"]:
        need(
            type(s) is dict
            and set(s) == {"z", "depth", "sha256"}
            and type(s["z"]) is int
            and s["z"] == z
            and type(s["depth"]) is int
            and 1 <= s["depth"] <= n - z
            and 4 * n * n * s["depth"] <= LIMIT,
            "slab gap, overlap or byte budget",
        )
        pin(s["sha256"])
        z += s["depth"]
    need(z == n, "incomplete z coverage")
    return d


@contextmanager
def stable(path):
    """Share read access only, including while the separate native process opens these names."""
    import msvcrt
    from ctypes import wintypes as w

    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    openfile = dll.CreateFileW
    openfile.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
    openfile.restype = w.HANDLE
    handle = openfile(str(path), 0x80000000, 1, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        dll.CloseHandle(w.HANDLE(handle))
        raise
    with os.fdopen(fd, "rb") as f:
        yield f


def paths(scan, build):
    return [scan.with_suffix(".geom"), scan.with_suffix(".f32"), *(build / n for n in NAMES)]


@contextmanager
def inputs(scan, build, d):
    with ExitStack() as stack:
        files = [stack.enter_context(stable(p)) for p in paths(scan, build)]
        need(os.fstat(files[1].fileno()).st_size == d["projections_bytes"], "projection length")
        for f, expected in zip(
            files,
            [d["geometry_sha256"], d["projections_sha256"], *(d["products"][n] for n in NAMES)],
            strict=True,
        ):
            need(
                hashlib.file_digest(f, "sha256").hexdigest() == expected,
                "geometry, projection or native product changed",
            )
        yield


def transfer(f, size, expected, dest=None, all_bytes=None):
    """Consume exactly one slab at the current offset; hash the very bytes written."""
    h = hashlib.sha256()
    left = size
    while left:
        b = f.read(min(left, STEP))
        need(bool(b), "short slab payload")
        h.update(b)
        if dest is not None:
            dest.write(b)
        if all_bytes is not None:
            all_bytes.update(b)
        left -= len(b)
    need(h.hexdigest() == expected, "slab payload SHA-256 mismatch")


def size(d, s):
    return 4 * d["n"] ** 2 * s["depth"]


def manifest(d, pp):
    return dict(
        format="CFDK-replayed-volume-v1",
        plan_sha256=pp,
        layout="float32 little-endian, z/y/x",
        shape_zyx=[d["n"]] * 3,
        volume_bytes=4 * d["n"] ** 3,
        volume_sha256=d["volume_sha256"],
        slabs=d["slabs"],
    )


def verify(plan_file, pp, delivery):
    d = plan(plan_file, pp)
    m = manifest(d, pp)
    need(bounded(delivery / "manifest.json", STEP) == encode(m), "different volume manifest")
    h = hashlib.sha256()
    with (delivery / "volume.f32").open("rb") as f:
        for s in d["slabs"]:
            transfer(f, size(d, s), s["sha256"], all_bytes=h)
        need(
            not f.read(1) and h.hexdigest() == d["volume_sha256"],
            "trailing or wrong assembled volume",
        )
    return m


def initialise(cache):
    marker = cache / "format.txt"
    try:
        cache.mkdir()
    except FileExistsError:
        need(
            cache.is_dir() and marker.is_file() and bounded(marker, 128) == MARKER,
            "not a private slab cache",
        )
    else:
        with marker.open("xb") as f:
            f.write(MARKER)
            f.flush()
            os.fsync(f.fileno())


@contextmanager
def writer(cache):
    # Separate lease permits independent durable BLOB commits without releasing the writer right.
    with closing(sqlite3.connect(cache / "writer.sqlite3", timeout=0)) as lease:
        try:
            lease.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as e:
            raise ValueError("another slab writer or unavailable lease") from e
        yield


def attach(db, pp, rows):
    need(
        db.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete",
        "unsupported journal mode",
    )
    db.execute("PRAGMA synchronous=FULL")
    with db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS identity(singleton INTEGER PRIMARY KEY CHECK(singleton=1), plan TEXT NOT NULL)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS slabs(z INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
        )
        db.execute("INSERT OR IGNORE INTO identity VALUES(1,?)", (pp,))
        need(
            db.execute("SELECT * FROM identity").fetchall() == [(1, pp)],
            "checkpoint belongs to a different plan",
        )
    saved = dict(db.execute("SELECT z, length(payload) FROM slabs"))
    need(
        all(z in rows and rows[z] == v for z, v in saved.items()),
        "unexpected checkpoint offset or length",
    )
    return saved


def run_native(scan, build, raw, d, s):
    from common import reconstruct

    n = d["n"]
    return reconstruct(
        scan,
        raw,
        n,
        d["fov_mm"],
        d["budget_mib"],
        d["batch"],
        build=build,
        roi=f"0,0,{s['z']},{n},{n},{s['depth']}",
    )


def attest(stats, d, s):
    n = d["n"]
    need(
        stats["n"] == n
        and stats["batch"] == d["batch"]
        and stats["roi_origin_xyz"] == [0, 0, s["z"]]
        and stats["output_shape_xyz"] == [n, n, s["depth"]]
        # The native diagnostic serializes a float32 spacing with nine digits.
        # Slab and volume payloads below still require exact SHA-256 equality.
        and math.isclose(stats["voxel_spacing_mm"], d["fov_mm"] / n, rel_tol=2e-7)
        and stats["redzones_verified"] is True
        and 0 < stats["peak_owned_bytes"] <= stats["budget_bytes"] == d["budget_mib"] << 20
        and stats["output_bytes"] == size(d, s),
        "native ROI, size or guard gate",
    )


def generate(plan_file, pp, scan, build, cache, dest, *, native=run_native, committed=None):
    plan_file, scan, build, cache, dest = [
        Path(p).resolve() for p in (plan_file, scan, build, cache, dest)
    ]
    d = plan(plan_file, pp)
    need(
        os.name == "nt"
        and not dest.exists()
        and dest.parent.is_dir()
        and not cache.is_relative_to(dest)
        and not dest.is_relative_to(cache)
        and all(
            not p.is_relative_to(s) and not s.is_relative_to(p)
            for p in (*paths(scan, build), plan_file)
            for s in (cache, dest)
        ),
        "fresh separate local Windows delivery and cache required",
    )
    used = computed = cached_bytes = 0
    metrics = []
    with inputs(scan, build, d):
        initialise(cache)
        with writer(cache), closing(sqlite3.connect(cache / "slabs.sqlite3", timeout=0)) as db:
            saved = attach(db, pp, {s["z"]: size(d, s) for s in d["slabs"]})
            for s in d["slabs"]:
                z, length = s["z"], size(d, s)
                if z in saved:
                    with db.blobopen("slabs", "payload", z, readonly=True) as blob:
                        transfer(blob, length, s["sha256"])
                    used += 1
                    cached_bytes += length
                    continue
                with tempfile.TemporaryDirectory(dir=cache, prefix=".compute-") as tmp:
                    raw = Path(tmp) / "slab.f32"
                    stats = native(scan, build, raw, d, s)
                    attest(stats, d, s)
                    with db, raw.open("rb") as f:
                        db.execute("INSERT INTO slabs VALUES(?,zeroblob(?))", (z, length))
                        with db.blobopen("slabs", "payload", z) as blob:
                            transfer(f, length, s["sha256"], blob)
                        need(not f.read(1), "trailing native slab")
                computed += 1
                metrics.append(
                    dict(
                        z=z,
                        depth=s["depth"],
                        gpu=stats["gpu"],
                        driver=stats["driver_api"],
                        peak_owned_bytes=stats["peak_owned_bytes"],
                    )
                )
                if committed is not None:
                    committed(computed)
            with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".assemble-") as tmp:
                stage = Path(tmp) / "delivery"
                stage.mkdir()
                h = hashlib.sha256()
                with (stage / "volume.f32").open("xb") as f:
                    for s in d["slabs"]:
                        with db.blobopen("slabs", "payload", s["z"], readonly=True) as blob:
                            transfer(blob, size(d, s), s["sha256"], dest=f, all_bytes=h)
                    need(h.hexdigest() == d["volume_sha256"], "assembled volume SHA-256 mismatch")
                    f.flush()
                    os.fsync(f.fileno())
                with (stage / "manifest.json").open("xb") as f:
                    f.write(encode(manifest(d, pp)))
                    f.flush()
                    os.fsync(f.fileno())
                os.rename(stage, dest)  # Windows no-overwrite, same parent/filesystem.
    return dict(
        reused=used,
        computed=computed,
        reused_bytes=cached_bytes,
        native_projection_bytes_per_invocation=d["projections_bytes"],
        native=metrics,
        volume_sha256=d["volume_sha256"],
        manifest_sha256=hashlib.sha256(encode(manifest(d, pp))).hexdigest(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    g = modes.add_parser("run")
    v = modes.add_parser("verify")
    for parser in (g, v):
        parser.add_argument("plan_file", type=Path)
        parser.add_argument("--plan-pin", required=True)
    for name in ("scan", "build", "cache", "dest"):
        g.add_argument(name, type=Path)
    v.add_argument("dest", type=Path)
    a = p.parse_args()
    result = (
        generate(a.plan_file, a.plan_pin, a.scan, a.build, a.cache, a.dest)
        if a.mode == "run"
        else verify(a.plan_file, a.plan_pin, a.dest)
    )
    print(json.dumps(result, indent=2))
