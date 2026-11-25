"""No CUDA: small asymmetric cube, transactional failures, source changes and portable reading."""

import hashlib
import sqlite3
from contextlib import closing

import pytest
import slab_replay as sr


def sha(b):
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def data(tmp_path):
    scan = tmp_path / "source"
    build = tmp_path / "build"
    build.mkdir()
    files = sr.paths(scan, build)
    for p, b in zip(files, (b"example geometry", b"abcdefghijkl", b"exe", b"ptx"), strict=True):
        p.write_bytes(b)
    # Each of the 3 z layers is distinguishable even after transposition or reversal.
    volume = bytes(range(108))
    names = sr.NAMES
    d = dict(
        format="CFDK-slab-replay-plan-v1",
        n=3,
        fov_mm=6.0,
        batch=1,
        budget_mib=3,
        geometry_sha256=sha(files[0].read_bytes()),
        projections_bytes=12,
        projections_sha256=sha(files[1].read_bytes()),
        products={n: sha((build / n).read_bytes()) for n in names},
        volume_sha256=sha(volume),
        slabs=[
            dict(z=0, depth=1, sha256=sha(volume[:36])),
            dict(z=1, depth=2, sha256=sha(volume[36:])),
        ],
    )
    p = tmp_path / "plan.json"
    p.write_bytes(sr.encode(d))
    return dict(
        p=p,
        pp=sha(p.read_bytes()),
        scan=scan,
        b=build,
        c=tmp_path / "cache",
        out=tmp_path / "result",
        d=d,
        volume=volume,
    )


def mock(scan, build, raw, d, s):
    z = s["z"]
    n = d["n"]
    count = sr.size(d, s)
    raw.write_bytes(bytes(range(108))[36 * z : 36 * z + count])
    return dict(
        gpu="test only",
        driver_api=1,
        n=n,
        batch=d["batch"],
        roi_origin_xyz=[0, 0, z],
        output_shape_xyz=[n, n, s["depth"]],
        voxel_spacing_mm=2,
        redzones_verified=True,
        peak_owned_bytes=16,
        budget_bytes=3 << 20,
        output_bytes=count,
    )


def run(data, **kw):
    t = data
    return sr.generate(
        t["p"], t["pp"], t["scan"], t["b"], t["c"], t["out"], native=kw.pop("native", mock), **kw
    )


def write(t):
    t["p"].write_bytes(sr.encode(t["d"]))
    t["pp"] = sha(t["p"].read_bytes())


def stop(n):
    raise InterruptedError("simulated callback failure after commit")


def test_recover_and_verify(data):
    t = data
    with pytest.raises(InterruptedError):
        run(t, committed=stop)
    assert not t["out"].exists()
    d = run(t)
    assert (d["reused"], d["computed"], d["reused_bytes"]) == (1, 1, 36)
    assert (t["out"] / "volume.f32").read_bytes() == t["volume"]
    assert sr.verify(t["p"], t["pp"], t["out"])["volume_sha256"] == sha(t["volume"])


@pytest.mark.parametrize(
    "field",
    [
        "n",
        "fov_mm",
        "batch",
        "budget_mib",
        "projections_bytes",
        "products",
        "volume_sha256",
        "format",
        "extra",
        "missing",
        "gap",
        "overlap",
        "short",
        "bool-depth",
        "oversize",
        "duplicate",
        "uppercase",
    ],
)
def test_reject_plan(data, field):
    t = data
    d = t["d"]
    if field in ("n", "batch", "budget_mib"):
        d[field] = True
    elif field == "fov_mm":
        d[field] = -1
    elif field == "projections_bytes":
        d[field] = 5
    elif field == "products":
        del d[field]["kernels.ptx"]
    elif field == "volume_sha256":
        d[field] = "abc"
    elif field == "format":
        d[field] = "x"
    elif field == "extra":
        d[field] = 0
    elif field == "missing":
        del d["slabs"]
    elif field == "gap":
        d["slabs"][1]["z"] = 2
    elif field == "overlap":
        d["slabs"][1]["z"] = 0
    elif field == "short":
        d["slabs"].pop()
    elif field == "bool-depth":
        d["slabs"][0]["depth"] = True
    elif field == "oversize":
        d["n"] = 4096
        d["slabs"] = [dict(z=0, depth=4096, sha256="0" * 64)]
    elif field == "uppercase":
        d["geometry_sha256"] = d["geometry_sha256"].upper()
    write(t)
    if field == "duplicate":
        t["p"].write_bytes(t["p"].read_bytes().replace(b'"n": 3', b'"n": 56, "n": 3'))
        t["pp"] = sha(t["p"].read_bytes())
    with pytest.raises(ValueError):
        sr.plan(t["p"], t["pp"])


@pytest.mark.parametrize("i", range(4))
def test_changed_sources_refused_on_resume(data, i):
    t = data
    with pytest.raises(InterruptedError):
        run(t, committed=stop)
    p = sr.paths(t["scan"], t["b"])[i]
    b = bytearray(p.read_bytes())
    b[-1] ^= 4
    p.write_bytes(b)
    with pytest.raises(ValueError, match="changed"):
        run(t)
    assert not t["out"].exists()


@pytest.mark.parametrize("i", range(4))
def test_native_cannot_race_pinned_files(data, i):
    t = data
    target = sr.paths(t["scan"], t["b"])[i]

    def probe(*args):
        with pytest.raises(PermissionError):
            with target.open("r+b") as f:
                f.write(b"X")
        with pytest.raises(PermissionError):
            target.rename(target.with_name("moved"))
        return mock(*args)

    assert run(t, native=probe)["computed"] == 2


@pytest.mark.parametrize("kind", ["length", "content", "foreign-row"])
def test_bad_checkpoint(data, kind):
    t = data
    with pytest.raises(InterruptedError):
        run(t, committed=stop)
    with closing(sqlite3.connect(t["c"] / "slabs.sqlite3")) as db, db:
        if kind == "foreign-row":
            db.execute("INSERT INTO slabs VALUES(10,?)", (b"",))
        else:
            db.execute(
                "UPDATE slabs SET payload=? WHERE z=0", (b"!" * (35 if kind == "length" else 36),)
            )
    with pytest.raises(ValueError, match="checkpoint|payload"):
        run(t)
    assert not t["out"].exists()


@pytest.mark.parametrize(
    "kind", ["wrong-final", "wrong-slab", "wrong-guard", "trailing", "truncated", "wrong-origin"]
)
def test_dirty_write_never_published(data, kind):
    t = data
    if kind == "wrong-final":
        t["d"]["volume_sha256"] = "0" * 64
        write(t)
    if kind == "wrong-slab":
        t["d"]["slabs"][0]["sha256"] = "0" * 64
        write(t)

    def broken(scan, build, path, d, s):
        m = mock(scan, build, path, d, s)
        if kind == "trailing":
            path.write_bytes(path.read_bytes() + b"!")
        if kind == "truncated":
            path.write_bytes(path.read_bytes()[:-1])
        if kind == "wrong-guard":
            m["redzones_verified"] = False
        if kind == "wrong-origin":
            m["roi_origin_xyz"][2] += 1
        return m

    with pytest.raises(ValueError):
        run(t, native=broken)
    assert not t["out"].exists()
    with closing(sqlite3.connect(t["c"] / "slabs.sqlite3")) as db:
        assert db.execute("select count(*) from slabs").fetchone()[0] == (
            2 if kind == "wrong-final" else 0
        )


def test_lease_stale_plan_and_existing_delivery(data):
    t = data
    with pytest.raises(InterruptedError):
        run(t, committed=stop)
    with sr.writer(t["c"]), pytest.raises(ValueError, match="another slab writer"):
        run(t)
    t["d"]["budget_mib"] = 4
    write(t)
    with pytest.raises(ValueError, match="different plan"):
        run(t)
    t["out"].mkdir()
    (t["out"] / "existing").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="fresh separate"):
        run(t)
    assert (t["out"] / "existing").read_bytes() == b"preserve"


@pytest.mark.parametrize("kind", ["unowned", "same", "nested", "wrong-pin"])
def test_separate_outputs(data, kind):
    t = data
    if kind == "unowned":
        t["c"].mkdir()
        (t["c"] / "important").write_bytes(b"keep")
    elif kind == "same":
        t["out"] = t["c"]
    elif kind == "nested":
        t["c"] = t["out"] / "inside"
    else:
        t["pp"] = "0" * 64
    with pytest.raises(ValueError):
        run(t)
    if kind == "unowned":
        assert (t["c"] / "important").read_bytes() == b"keep"
    assert not t["out"].exists()


@pytest.mark.parametrize("kind", ["manifest", "volume", "trailing", "truncated"])
def test_changed_delivery(data, kind):
    t = data
    run(t)
    p = t["out"] / ("manifest.json" if kind == "manifest" else "volume.f32")
    b = p.read_bytes()
    p.write_bytes(
        b + b"!"
        if kind == "trailing"
        else b[:-1]
        if kind == "truncated"
        else bytes([b[0] ^ 3]) + b[1:]
    )
    with pytest.raises(ValueError):
        sr.verify(t["p"], t["pp"], t["out"])
