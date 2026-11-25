"""Parallelizable work must not change the externally pinned slab, transaction or delivery unit."""

import copy
import hashlib
import itertools
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import grouped_replay as g
import qualify_grouped_replay as q
from grouped_replay import sr, old


def plan():
    return sr.plan(g.FIXTURE, g.PLAN)


def hashes(raw):
    return hashlib.sha256(raw).hexdigest()


def tiny(tmp):
    d = plan()
    d["n"] = 3
    d["fov_mm"] = 12.0
    ds = [(0, 1), (1, 2)]
    payloads = [bytes((i * 13 + z) % 256 for i in range(36 * depth)) for z, depth in ds]
    slabs = [
        dict(z=z, depth=dep, sha256=hashes(raw))
        for (z, dep), raw in zip(ds, payloads, strict=True)
    ]
    d["slabs"] = slabs
    raw = tmp / "packed"
    head = g.header(d, slabs)
    raw.write_bytes(head + b"".join(payloads))
    r = json.loads((g.M / "grouped-replay-pilot-001/run.json").read_text())["groups"][0]["native"]
    r.update(
        n=3, voxel_spacing_mm=4.0, output_bytes=raw.stat().st_size, launches_bp=90, regions=[]
    )
    offset = len(head)
    for s, p in zip(slabs, payloads, strict=True):
        r["regions"].append(
            dict(
                origin_xyz=[0, 0, s["z"]],
                shape_xyz=[3, 3, s["depth"]],
                offset_bytes=offset,
                size_bytes=len(p),
            )
        )
        offset += len(p)
    return d, slabs, raw, r, payloads


@pytest.mark.parametrize("cap", [1, 2, 3, 4, 5])
def test_every_sparse_subset(cap):
    d = plan()
    ss = d["slabs"]
    for mask in itertools.product((False, True), repeat=5):
        saved = {s["z"] for s, k in zip(ss, mask, strict=True) if k}
        want = [s for s in ss if s["z"] not in saved]
        groups = g.schedule(d, saved, cap)
        assert (
            [s for group in groups for s in group] == want
            and all(0 < len(group) <= cap for group in groups)
            and len(groups) == (len(want) + cap - 1) // cap
        )


@pytest.mark.parametrize("cap", [0, 6, True, 1.5, -1])
def test_bad_group(cap):
    with pytest.raises(ValueError):
        g.schedule(plan(), [], cap)


@pytest.mark.parametrize(
    "change",
    [
        "header",
        "shape",
        "order",
        "payload_first",
        "payload_last",
        "truncated",
        "extra",
        "hardlink",
        "symlink",
        "guards",
        "budget",
        "name",
        "passes",
        "fft",
        "projections",
        "bytes",
        "alias_integer",
        "offset",
        "extent",
        "kernel",
        "peak",
        "pinned",
    ],
)
def test_no_invalid_group_commits(tmp_path, change):
    d, slabs, raw, r, parts = tiny(tmp_path)
    b = raw.read_bytes()
    hd = len(g.header(d, slabs))
    if change in ("header", "shape", "payload_first", "payload_last"):
        off = {"header": 0, "shape": 20, "payload_first": hd, "payload_last": len(b) - 1}[change]
        raw.write_bytes(b[:off] + bytes([b[off] ^ 1]) + b[off + 1 :])
    if change == "order":
        slabs = slabs[::-1]
    if change == "truncated":
        raw.write_bytes(b[:-1])
    if change == "extra":
        raw.write_bytes(b + b"x")
    if change == "hardlink":
        os.link(raw, tmp_path / "copy")
    if change == "symlink":
        link = tmp_path / "sym"
        link.symlink_to(raw)
        raw = link
    if change == "guards":
        r["redzones_verified"] = 1
    if change == "budget":
        r["budget_bytes"] += 1
    if change == "name":
        r["gpu"] = "other"
    if change == "passes":
        r["projection_passes"] = 2
    if change == "fft":
        r["filter_batches"] = 90
    if change == "projections":
        r["projection_bytes_read"] -= 4
    if change == "bytes":
        r["output_bytes"] -= 1
    if change == "alias_integer":
        r["regions"][0]["origin_xyz"][0] = False
    if change == "offset":
        r["regions"][1]["offset_bytes"] -= 4
    if change == "extent":
        r["regions"][0]["size_bytes"] += 4
    if change == "kernel":
        r["launches_bp"] -= 1
    if change == "peak":
        r["peak_owned_bytes"] = r["budget_bytes"] + 1
    if change == "pinned":
        r["host_pinned_bytes"] = 0
    with closing(sqlite3.connect(tmp_path / "slabs.sqlite3")) as db:
        sr.attach(db, g.PLAN, {s["z"]: sr.size(d, s) for s in slabs})
        with pytest.raises((ValueError, OSError)):
            g.fill(db, d, slabs, raw, r, None, 0)
        assert not db.execute("select * from slabs").fetchall() and raw.exists()


def test_verified_bytes_committed_individually(tmp_path):
    d, slabs, raw, r, parts = tiny(tmp_path)
    seen = []
    with closing(sqlite3.connect(tmp_path / "slabs.sqlite3")) as db:
        sr.attach(db, g.PLAN, {s["z"]: sr.size(d, s) for s in slabs})
        g.fill(db, d, slabs, raw, r, lambda n: seen.append((n, db.in_transaction)), 7)
        assert (
            not raw.exists()
            and seen == [(8, False), (9, False)]
            and db.execute("select payload from slabs order by z").fetchall()
            == [(p,) for p in parts]
        )


@pytest.mark.parametrize("start", ["serial", "grouped"])
def test_real_cache_interoperability(start, tmp_path):
    class Stop(RuntimeError):
        pass

    def cut(n):
        if n == 2:
            raise Stop

    d = plan()
    wd = tmp_path / "job"
    wd.mkdir(mode=0o700)
    cache = wd / "cache"
    dst = wd / "delivered"

    def serial(callback):
        return old.generate(g.FIXTURE, g.PLAN, g.SCAN, g.BUILD, cache, dst, committed=callback)

    def grouped(callback):
        return g.issue(cache, dst, committed=callback)

    run, end = (serial, grouped) if start == "serial" else (grouped, serial)
    with pytest.raises(Stop):
        run(cut)
    assert not dst.exists()
    r = end(None)
    assert r["computed"] == 3 and sr.verify(g.FIXTURE, g.PLAN, dst) == sr.manifest(d, g.PLAN)
    assert (r["reused_z"] == [0, 73]) if start == "serial" else (r["reused"] == 2)


def test_corrupt_reused_blob_refuses_before_worker(tmp_path):
    wd = tmp_path / "owner"
    wd.mkdir(mode=0o700)
    cache = wd / "cache"
    old.validate_cache(cache)
    d = plan()
    s = d["slabs"][0]
    with closing(sqlite3.connect(cache / "slabs.sqlite3")) as db:
        sr.attach(db, g.PLAN, {s["z"]: sr.size(d, s)})
        with db:
            db.execute("insert into slabs values(?,zeroblob(?))", (s["z"], sr.size(d, s)))

    def forbidden(*a):
        pytest.fail("dispatched CUDA before authenticating a corrupt reused slab")

    with pytest.raises(ValueError, match="payload SHA"):
        g.issue(cache, wd / "dest", runner=forbidden)
    assert not (wd / "dest").exists()


RAW = "1546c968ae06018b1a9b6b687da760332cbb2613bedc3438a6814a12ea7d7946"
import review_grouped_replay as review


def recorded():
    return review.load(g.M / "grouped-replay-qualified-001/qualification.json", RAW)


def test_reopen_dual_verdict():
    r = recorded()
    v = review.check(r, plan())
    assert (
        v["decision"] == q.decision(r["cold"], plan()) == r["decision"]
        and v["decision"]["wins"] == 6
        and v["decision"]["gate"] is True
        and v["nonprefix_missing"] == [73, 283]
        and v["cached_workers"] == 0
    )


@pytest.mark.parametrize(
    "fault",
    [
        "order",
        "warm",
        "worker_count",
        "timing",
        "roi_offset",
        "volume",
        "reused",
        "unpublished",
        "partial",
        "missing_kill",
        "fully_cached",
        "wrong_group",
        "peak",
        "decision",
        "reference",
        "acquisition",
        "serial_shape",
        "bounds",
        "hole",
    ],
)
def test_detached_refuses_tampering(fault):
    r = recorded()
    cold = r["cold"]
    a = next(x for x in cold if x["arm"] == "grouped")
    n = a["report"]["groups"][0]["native"]
    if fault == "order":
        cold.reverse()
    if fault == "warm":
        r["warm"].pop()
    if fault == "worker_count":
        a["workers"] = True
    if fault == "timing":
        a["end_to_end_ms"] = float("inf")
    if fault == "roi_offset":
        n["regions"][1]["offset_bytes"] += 4
    if fault == "volume":
        a["report"]["volume_sha256"] = "e" * 64
    if fault == "reused":
        r["crashes"]["inside"]["resume"]["reused_bytes"] = 0
        r["crashes"]["inside"]["resume"]["report"]["reused_bytes"] = 0
    if fault == "unpublished":
        r["crashes"]["between"]["unpublished"] = False
    if fault == "partial":
        r["crashes"]["inside"]["prefix"].append([174, 1 << 20])
    if fault == "missing_kill":
        del r["crashes"]["between"]
    if fault == "fully_cached":
        r["fully_cached"]["workers"] = 1
    if fault == "wrong_group":
        a["report"]["groups"][0]["z"].reverse()
    if fault == "peak":
        a["peak_owned_bytes"] = 769 << 20
    if fault == "decision":
        r["decision"]["reduction"] = 0
    if fault == "reference":
        a["report"]["parents"]["astra"] = "0" * 64
    if fault == "acquisition":
        n["projection_bytes_read"] -= 4
    if fault == "serial_shape":
        next(x for x in cold if x["arm"] == "serial")["report"]["native"][0]["depth"] = 1
    if fault == "bounds":
        n["redzones_verified"] = 1
    if fault == "hole":
        r["non_prefix"]["computed_z"] = [174, 283]
    with pytest.raises(ValueError):
        review.check(r, plan())


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_reject_ambiguous_json(tmp_path, raw):
    f = tmp_path / "doc"
    f.write_bytes(raw)
    with pytest.raises(ValueError):
        review.load(f, hashes(raw))
    with pytest.raises(ValueError):
        review.load(f, "f" * 64)
