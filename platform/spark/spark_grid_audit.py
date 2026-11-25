"""Audit a user-pinned direct grid pack against fresh ASTRA reconstructions before publishing an independent witness pack."""

import argparse, fcntl, json, math, os, struct
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import bench_astra_lease as bench
import grid_oracle as g
import spark_multigrid as s
from grouped_replay import M, MIB, sha, need
from sealed_inputs import snapshot
from spark_publication import fresh, staging, publish
from astra_projection_lease import projection_lease
from residual_witness import pin

BENCH = "38fb339c6aff13c151f7fe4c395c9a78ee6815d7f5e49a6fc97f6c50bcdd4320"


def admission(dims, grids):
    a, v, u = dims
    need(
        3 <= a <= 4096 and 2 <= v <= 2048 and 2 <= u <= 2048 and 4 * a * v * u <= 256 * MIB,
        "scan exceeds oracle envelope",
    )
    shapes = [g.enclosure(n, b)[0] for n, f, b in grids]
    vox = [math.prod(s) for s in shapes]
    out = 4 * sum(math.prod(b[3:]) for _, _, b in grids)
    need(
        max(vox) * 4 <= 256 * MIB
        and sum(vox) * a <= 100_000_000_000
        and out <= 256 * MIB
        and sum(b[-1] for _, _, b in grids) <= 6000,
        "requested centered witnesses exceed oracle envelope",
    )
    return dict(
        scan_bytes=4 * a * v * u,
        largest_centered_voxels=max(vox),
        enclosed_voxel_views=sum(vox) * a,
        output_payload_bytes=out,
    )


def route(d, grids, receipt):
    if (
        d["geometry_sha256"] == receipt["inputs"]["full-corrected.geom"]
        and d["projections_sha256"] == receipt["inputs"]["full-corrected.f32"]
        and grids == g.GRIDS
    ):
        return "borrow"
    return "copy"


def qualification():
    oracle, base = bench.pinned()
    b = s.strict.load(M / "astra-grid-lease-001/benchmark.json", BENCH)
    need(
        b["oracle"] == bench.ORACLE
        and b["source"]
        == {
            n: sha(bench.SRC / n)
            for n in ("astra_projection_lease.py", "bench_astra_lease.py", "grid_oracle.py")
        }
        and s.eq(b["summary"], bench.compare(b["rows"]))
        and b["summary"]["selected"] is True,
        "unqualified ASTRA lease",
    )
    return oracle, base


def audit(request, request_pin, geom, scan, native, native_pin, out):
    import astra

    out = fresh(out)
    pin(native_pin)
    pin(request_pin)
    need(
        all(
            not out.is_relative_to(Path(p).resolve())
            and not Path(p).resolve().is_relative_to(out)
            for p in (request, geom, scan, native)
        ),
        "output aliases input",
    )
    oracle, base = qualification()
    env = g.libcheck(astra, base)
    with ExitStack() as st, staging(out, ".checked-grid-oracle-") as td:
        stage = Path(td) / "ready"
        stage.mkdir()
        read = lambda p, k, limit, size=None: Path(
            "/proc/self/fd/" + str(st.enter_context(snapshot(p, k, limit=limit, exact=size)))
        )
        r = s.strict.load(read(request, request_pin, 65536), request_pin)
        grids = s.validate_recipe(r)
        gp = read(geom, r["geometry_sha256"], 4096)
        dims, params = g.geometry(gp.read_bytes())
        need(r["batch"] <= dims[0], "batch exceeds views")
        plan = admission(dims, grids)
        npth = read(
            native,
            native_pin,
            256 * MIB + 1024,
            12 + 36 * len(grids) + plan["output_payload_bytes"],
        )
        meta = g.verify(npth, native_pin)
        s.layout(meta, grids)
        spr = read(scan, r["projections_sha256"], 256 * MIB, plan["scan_bytes"])
        views = np.memmap(spr, dtype="<f4", mode="r", shape=dims)
        need(all(np.isfinite(view).all() for view in views), "nonfinite projection")
        pg = g.proj(astra, dims, params)
        mode = route(r, grids, oracle)
        ref = stage / "astra.mg"
        rows = []
        with projection_lease(astra, pg, views, mode) as pid, ref.open("xb") as f:
            f.write(struct.pack("<8sI", b"CFDKMG01", len(grids)))
            for n, fov, b in grids:
                f.write(struct.pack("<Iff6I", n, fov, g.fp32(fov / n), *b))
            for i, (v, (n, fov, b)) in enumerate(zip(meta["regions"], grids, strict=True)):
                y, w = g.reconstruct(astra, pid, n, fov, b)
                x = np.memmap(
                    npth,
                    dtype="<f4",
                    mode="r",
                    offset=v["offset_bytes"],
                    shape=tuple(reversed(b[3:])),
                )
                row = g.score(x, y, b[:3])
                need(
                    g.passes(row, g.LIMIT),
                    f"grid {i} at parent {n} fails independent comparison: rel_l2={row['rel_l2']:.9g}, max_abs={row['max_abs']:.9g}; no witness issued",
                )
                y.tofile(f)
                rows.append(
                    dict(parent_n=n, box=list(b), fov_mm=fov, witness=w, native_vs_astra=row)
                )
        rd = g.verify(ref, sha(ref))
        s.layout(rd, grids)
        if mode == "borrow":
            need(rd == oracle["oracle_pack"], "preset reference drift")
        doc = dict(
            format="CFDK-Spark-user-grid-audit-v1",
            request_sha256=request_pin,
            geometry_sha256=r["geometry_sha256"],
            projections_sha256=r["projections_sha256"],
            native_pack_sha256=native_pin,
            oracle_pack=rd,
            astra=env,
            projection_ownership=mode,
            qualification_parents=dict(oracle=bench.ORACLE, lease=BENCH),
            issuer_sha256=sha(Path(__file__)),
            admission=plan,
            limits=dict(native_rel_l2=g.LIMIT[0], native_max_abs=g.LIMIT[1]),
            regions=rows,
            passed=True,
            scope="Exact pinned arrays, independent full-voxel comparison, no registration, gain fitting or physical dimensional claim. Unknown acquisitions may legitimately fail these fixed gates.",
        )
        (stage / "qualification.json").write_text(
            json.dumps(doc, indent=2, allow_nan=False) + "\n"
        )
        publish(stage, out)
        return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "request_pin", "geom", "scan", "native", "native_pin", "out"):
        p.add_argument(name, type=str if name.endswith("pin") else Path)
    a = p.parse_args()
    with (M / "gpu.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        d = audit(**vars(a))
        print(
            json.dumps(
                dict(ownership=d["projection_ownership"], pack=d["oracle_pack"]["pack_sha256"])
            )
        )
