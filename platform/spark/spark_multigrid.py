"""Issue a direct mixed-grid FDK pack on GB10. For new acquisitions each grid must match a fresh independent frozen-worker result."""

import argparse, fcntl, hashlib, json, math, os, re, tempfile
from contextlib import ExitStack
from pathlib import Path
import qualify_multigrid as q
import review_multigrid as v
import review_budget_frontier as strict
from qualify_multigrid import g, M, need, sha, snapshot, staging, publish, fresh, MIB

PROFILE = "73eb45ab5c65165132b7038e5d196289373d90eb9892ec6cc6157bff8eaaeee8"
PACK = "806aaf2511308d5ebf105921452bfbc05827c6db261be2e1b0266e6d62b1f530"
QDIR = M / "multigrid-qualification-001"
BDIR = M / "multigrid-build-002"


def validate_recipe(r):
    need(
        isinstance(r, dict)
        and set(r)
        == {"format", "geometry_sha256", "projections_sha256", "batch", "cap_mib", "grids"}
        and r["format"] == "CFDK-multigrid-request-v1",
        "request schema",
    )
    need(
        all(
            type(r[k]) is str and re.fullmatch("[0-9a-f]{64}", r[k])
            for k in ("geometry_sha256", "projections_sha256")
        ),
        "input pins",
    )
    need(
        type(r["batch"]) is int
        and 1 <= r["batch"] <= 256
        and type(r["cap_mib"]) is int
        and 1 <= r["cap_mib"] <= 2048,
        "batch or cap",
    )
    items = r["grids"]
    need(type(items) is list and 1 <= len(items) <= 16, "grid count")
    grids = []
    total = 0
    for t in items:
        need(
            isinstance(t, dict)
            and set(t) == {"n", "fov_mm", "box_xyz"}
            and type(t["n"]) is int
            and 2 <= t["n"] <= 4096,
            "grid dimension",
        )
        n = t["n"]
        f = t["fov_mm"]
        need(type(f) in (float, int) and 1e-6 <= f <= 1e6, "field of view")
        f = v.fp32(f)
        b = t["box_xyz"]
        need(
            type(b) is list
            and len(b) == 6
            and all(type(x) is int for x in b)
            and all(a >= 0 and h > 0 and a + h <= n for a, h in zip(b[:3], b[3:], strict=True)),
            "grid bounds",
        )
        total += 4 * math.prod(b[3:])
        grids.append((n, f, tuple(b)))
    need(total <= 1024 * MIB, "output payload exceeds 1 GiB")
    return tuple(grids)


def eq(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def layout(meta, grids):
    need(meta["format"] == "CFDKMG01" and len(meta["regions"]) == len(grids), "payload count")
    cur = 12 + 36 * len(grids)
    for r, (n, f, b) in zip(meta["regions"], grids, strict=True):
        pitch = v.fp32(v.fp32(f) / n)
        sz = 4 * math.prod(b[3:])
        des = dict(
            n=n,
            fov_mm=v.fp32(f),
            pitch_mm=pitch,
            origin_xyz=list(b[:3]),
            shape_xyz=list(b[3:]),
            offset_bytes=cur,
            size_bytes=sz,
            first_voxel_center_mm=[(x - (n - 1) * 0.5) * pitch for x in b[:3]],
        )
        need(
            all(eq(r[k], y) for k, y in des.items())
            and re.fullmatch("[a-f0-9]{64}", r["sha256"]),
            "payload placement or digest",
        )
        cur += sz


def certificate():
    d = strict.load(g.FIXTURE, q.PLAN)
    p = strict.load(QDIR / "qualification.json", PROFILE)
    b = strict.load(BDIR / "build.json", q.BUILD)
    need(
        b["builder"] == sha(q.HERE / "build_multigrid.py")
        and b["header"] == sha(q.HERE / "fdk/multigrid.hpp")
        and b["builds"][0] == b["builds"][1]
        and b["ptx"] == q.PTX
        and b["original"] == q.BASE,
        "incorrect build",
    )
    need(
        p["source"] == {a.name: sha(a) for a in (Path(q.__file__), Path(v.__file__))}
        and eq(p["grids"], q.GRIDS)
        and p["parents"] == dict(plan=q.PLAN, build=q.BUILD, original=q.NATIVE, astra=q.ASTRA)
        and p["geometry"] == d["geometry_sha256"]
        and p["projection"] == d["projections_sha256"]
        and p["grid_spec_sha256"] == hashlib.sha256(q.spec(q.GRIDS)).hexdigest()
        and p["pack"]["pack_sha256"] == PACK
        and p["budget_mib"] == 256
        and p["batch"] == 16,
        "wrong acquisition",
    )
    layout(p["pack"], q.GRIDS)
    rows = p["samples"]
    warm = p["warmups"]
    need(
        [(r["block"], r["arm"]) for r in warm] == [(-1, "serial"), (-1, "shared")], "warmup order"
    )
    keys = [r["sha256"] for r in p["pack"]["regions"]]
    need(keys[1] == p["reference_512_crop_sha256"], "independent 512 full-volume crop")
    for r in warm + rows:
        shared = r["arm"] == "shared"
        ns = r["native"]
        need(
            r["arm"] in ("shared", "serial")
            and len(ns) == (1 if shared else 3)
            and r["per_grid_sha256"] == keys,
            "per-grid baseline",
        )
        for i, n in enumerate(ns):
            q.guard(
                n,
                (720, d["projections_bytes"]),
                256,
                16,
                q.GRIDS if shared else [q.GRIDS[i]],
                shared,
            )
        ts = r["process_times_ms"]
        need(
            len(ts) == len(ns)
            and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in ts)
            and sum(ts) == r["native_process_ms"]
            and r["scan_bytes"] == len(ns) * d["projections_bytes"]
            and r["peak_owned_bytes"] == max(n["peak_owned_bytes"] for n in ns),
            "trial accounting",
        )
    need(
        eq(p["decision"], q.compare(rows)) and p["decision"]["gate"] is True,
        "performance gate failed",
    )
    return d, b, p


def preset(d):
    return dict(
        format="CFDK-multigrid-request-v1",
        geometry_sha256=d["geometry_sha256"],
        projections_sha256=d["projections_sha256"],
        batch=16,
        cap_mib=256,
        grids=[dict(n=n, fov_mm=f, box_xyz=list(b)) for n, f, b in q.GRIDS],
    )


def issue(out, *, user=None):
    out = fresh(out)
    d, build, profile = certificate()
    base = g.ROOT / "build/spark"
    ref = M / "reference"
    if user is None:
        recipe = preset(d)
        geo, pr = ref / "full-corrected.geom", ref / "full-corrected.f32"
        recipe_pin = None
    else:
        request, recipe_pin, geo, pr = user
        recipe = strict.load(request, recipe_pin)
    grids = validate_recipe(recipe)
    batch = recipe["batch"]
    cap = recipe["cap_mib"]
    all_inputs = [geo, pr, BDIR / "multigrid", base / "kernels.ptx", base / "cone-fdk"] + (
        [user[0]] if user else []
    )
    need(
        all(
            not out.is_relative_to(Path(p).resolve())
            and not Path(p).resolve().is_relative_to(out)
            for p in all_inputs
        ),
        "input aliases output",
    )
    spec_bytes = q.spec(grids)
    spec_pin = hashlib.sha256(spec_bytes).hexdigest()
    timings = []
    with ExitStack() as st, staging(out, ".multigrid-issue-") as td:
        stage = Path(td) / "ready"
        wd = Path(td).resolve() / "scratch"
        stage.mkdir()
        wd.mkdir(mode=0o700)

        def hold(path, pin, limit, ex=False, exact=None):
            return st.enter_context(snapshot(path, pin, limit=limit, executable=ex, exact=exact))

        loc = lambda fd: f"/proc/self/fd/{fd}"
        ge = hold(geo, recipe["geometry_sha256"], 4096)
        header = os.pread(ge, 4096, 0).decode("ascii").split()
        need(
            len(header) == 10
            and header[0] == "CFDK1"
            and all(re.fullmatch("[0-9]+", i) for i in header[1:4]),
            "acquisition header",
        )
        views, h, w = map(int, header[1:4])
        size = 4 * views * h * w
        need(
            3 <= views <= 65535
            and 2 <= h <= 4096
            and 2 <= w <= 4096
            and size <= 1024 * MIB
            and batch <= views,
            "acquisition extent or batch",
        )
        src = hold(pr, recipe["projections_sha256"], 1024 * MIB, exact=size)
        ptx = hold(base / "kernels.ptx", q.PTX, 4 * MIB)
        exe = hold(BDIR / "multigrid", build["builds"][0]["multigrid"], 8 * MIB, True)
        sf = wd / "grids.txt"
        sf.write_bytes(spec_bytes)
        gf = hold(sf, spec_pin, 16384)
        cmd = [
            "--geom",
            loc(ge),
            "--input",
            loc(src),
            "--ptx",
            loc(ptx),
            "--budget",
            str(cap),
            "--batch",
            str(batch),
        ]
        fds = (ge, src, ptx)
        pack = stage / "volume.mg"
        native, ms = q.execute(
            [loc(exe), *cmd, "--grids", loc(gf), "--output", pack], (*fds, exe, gf)
        )
        q.guard(native, (views, size), cap, batch, grids, True)
        with pack.open("rb") as f:
            meta = v.read(f)
        layout(meta, grids)
        mode = "pinned-walnut"
        if user is None:
            need(
                eq(meta, profile["pack"]), "preset differs from frozen independently checked pack"
            )
        else:
            mode = "fresh-original-per-grid"
            og = hold(base / "cone-fdk", q.BASE, 8 * MIB, True)
            for i, ((n, f, box), cell) in enumerate(zip(grids, meta["regions"], strict=True)):
                raw = wd / "original.f32"
                need(not os.path.lexists(raw), "temporary collision")
                ac = [*cmd, "--n", str(n), "--fov", f"{f:.9g}", "--output", raw]
                if box != (0, 0, 0, n, n, n):
                    ac += ["--roi", ",".join(map(str, box))]
                nr, tm = q.execute([loc(og), *ac], (*fds, og))
                q.guard(nr, (views, size), cap, batch, [(n, f, box)], False)
                need(
                    raw.stat().st_size == cell["size_bytes"] and sha(raw) == cell["sha256"],
                    "fresh standalone grid disagrees, no delivery",
                )
                timings.append(dict(index=i, sha256=cell["sha256"], native=nr, elapsed_ms=tm))
                raw.unlink()
        receipt = dict(
            format="CFDK-GB10-direct-grid-delivery-v1",
            mode=mode,
            issuer_sha256=sha(__file__),
            profile=PROFILE,
            build=q.BUILD,
            geometry_sha256=recipe["geometry_sha256"],
            projections_sha256=recipe["projections_sha256"],
            request_sha256=recipe_pin,
            native_grid_spec_sha256=spec_pin,
            native=native,
            native_process_ms=ms,
            independent=timings,
            pack=meta,
            scope="Direct FDK; pitch is nominal scanner-local sampling, not measured resolution. External input pins are byte identities, not scanner calibration or authenticity of origin. New grids require exact fresh original comparisons; their total delivery time is not the three-grid preset benchmark.",
        )
        (stage / "delivery.json").write_text(json.dumps(receipt, indent=2) + "\n")
        publish(stage, out)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("walnut")
    s.add_argument("out", type=Path)
    c = sub.add_parser("checked")
    c.add_argument("request", type=Path)
    c.add_argument("request_sha256")
    c.add_argument("geometry", type=Path)
    c.add_argument("projections", type=Path)
    c.add_argument("out", type=Path)
    a = parser.parse_args()
    with (M / "gpu.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = issue(
            a.out,
            user=None
            if a.command == "walnut"
            else (a.request, a.request_sha256, a.geometry, a.projections),
        )
        print(
            json.dumps(
                dict(
                    mode=r["mode"],
                    pack_sha256=r["pack"]["pack_sha256"],
                    regions=len(r["pack"]["regions"]),
                )
            )
        )
