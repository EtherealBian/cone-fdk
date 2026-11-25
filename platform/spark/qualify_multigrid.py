"""Direct multiresolution FDK from one scan: balanced comparison with three independent frozen native runs."""

import argparse, hashlib, json, math, os, stat, subprocess, tempfile, time
from contextlib import ExitStack
from pathlib import Path
from statistics import median
import grouped_replay as g
import review_multigrid as review
from build_multigrid import PARENT, HERE
from tune_backproject import BASE, PTX, NATIVE, ASTRA, PLAN, json_checked
from grouped_replay import M, MIB, need, sha
from spark_publication import fresh, staging, publish
from sealed_inputs import snapshot

BUILD = "e83e4da1a93e74752eb671bac813c6042f5054ae21edcba46e06cbdc9553f381"
GRIDS = (
    (192, 40.0, (0, 0, 0, 192, 192, 192)),
    (512, 40.0, (181, 161, 173, 128, 120, 99)),
    (1024, 32.0, (377, 397, 425, 193, 159, 131)),
)


def spec(grids):
    return (
        "CFDK-GRIDS-1\n"
        + "".join(f"{n} {fov:.9g} " + ",".join(map(str, box)) + "\n" for n, fov, box in grids)
    ).encode()


def crop(fd, n, box):
    h = hashlib.sha256()
    x, y, z, nx, ny, nz = box
    for k in range(z, z + nz):
        for j in range(y, y + ny):
            v = os.pread(fd, 4 * nx, 4 * ((k * n + j) * n + x))
            need(len(v) == 4 * nx, "short volume crop")
            h.update(v)
    return h.hexdigest()


def guard(r, scan, cap, batch, grids, packed):
    views, projection_bytes = scan
    count = len(grids)
    checks = dict(
        gpu="NVIDIA GB10",
        driver_api=13000,
        cufft_version=12000,
        views=views,
        batch=batch,
        requested_batch=batch,
        budget_bytes=cap * MIB,
        effective_budget_bytes=cap * MIB,
        redzones_verified=True,
        projection_passes=1,
        projection_bytes_read=projection_bytes,
        filter_batches=(views + batch - 1) // batch,
        launches_bp=count * ((views + batch - 1) // batch),
        feasible_plans_scored=[dict(batch=batch, passes=1)],
        planner="batch-first",
    )
    eq = lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    need(
        all(k in r and eq(r[k], v) for k, v in checks.items()), "native scan/plan/guard mismatch"
    )
    need(
        type(r["peak_owned_bytes"]) is int
        and 0 < r["peak_owned_bytes"] <= cap * MIB
        and type(r["fft_workspace_bytes"]) is int
        and r["fft_workspace_bytes"] >= 0,
        "memory gate",
    )
    offs = 12 + 36 * count if packed else 0
    if packed:
        need(
            r["format"] == "CFDKMG01"
            and len(r["regions"]) == count
            and r["first_grid_n"] == grids[0][0]
            and math.isclose(
                r["first_voxel_spacing_mm"], review.fp32(grids[0][1] / grids[0][0]), rel_tol=1e-8
            ),
            "bad pack identifier",
        )
    for i, (n, fov, box) in enumerate(grids):
        size = 4 * math.prod(box[3:])
        p = review.fp32(review.fp32(fov) / n)
        if packed:
            row = r["regions"][i]
            need(
                row["n"] == n
                and math.isclose(row["fov_mm"], fov, rel_tol=1e-8)
                and math.isclose(row["pitch_mm"], p, rel_tol=1e-8)
                and eq(row["origin_xyz"], list(box[:3]))
                and eq(row["shape_xyz"], list(box[3:]))
                and row["offset_bytes"] == offs
                and row["size_bytes"] == size,
                "grid placement",
            )
        else:
            need(
                count == 1
                and r["n"] == n
                and r["slabs"] == 1
                and r["slab"] == box[-1]
                and eq(r["roi_origin_xyz"], list(box[:3]))
                and eq(r["output_shape_xyz"], list(box[3:]))
                and math.isclose(r["voxel_spacing_mm"], p, rel_tol=1e-8),
                "baseline placement",
            )
        offs += size
    need(r["output_bytes"] == offs, "output bytes")


def execute(cmd, fds):
    t = time.monotonic_ns()
    p = subprocess.run(list(map(str, cmd)), pass_fds=tuple(fds), capture_output=True, timeout=120)
    dt = (time.monotonic_ns() - t) / 1e6
    need(
        not p.returncode and not p.stderr,
        "native failure: " + p.stderr.decode(errors="replace")[-600:],
    )
    return json.loads(p.stdout), dt


def compare(rows):
    order = [
        (i, k)
        for i in range(6)
        for k in (("serial", "shared") if i % 2 == 0 else ("shared", "serial"))
    ]
    need([(r["block"], r["arm"]) for r in rows] == order, "incomplete paired design")
    a, b = ([r["native_process_ms"] for r in rows if r["arm"] == k] for k in ("serial", "shared"))
    need(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in a + b), "bad timer")
    x, y = map(median, (a, b))
    wins = sum(v < u for u, v in zip(a, b, strict=True))
    return dict(
        serial_ms=x,
        shared_ms=y,
        reduction=1 - y / x,
        wins=wins,
        pairs=6,
        gate=wins >= 5 and y <= 0.95 * x,
    )


def run(out):
    out = fresh(out)
    bld = M / "multigrid-build-002"
    base = g.ROOT / "build/spark"
    ref = M / "reference"
    d = json_checked(g.FIXTURE, PLAN)
    orig = json_checked(M / "walnut-gb10-001/result.json", NATIVE)
    astra = json_checked(M / "walnut-astra-gb10-001/result.json", ASTRA)
    br = json_checked(bld / "build.json", BUILD)
    need(
        br["builder"] == sha(HERE / "build_multigrid.py")
        and br["header"] == sha(HERE / "fdk/multigrid.hpp")
        and br["builds"][0] == br["builds"][1]
        and br["parent"] == PARENT
        and br["ptx"] == PTX
        and br["original"] == BASE
        and orig["input_sha256"] == astra["input_sha256"]
        and astra["passed"] is True
        and orig["output_sha256"] == [d["volume_sha256"]] * 2,
        "invalid provenance",
    )
    scan = (720, d["projections_bytes"])
    cap = 256
    batch = 16
    with ExitStack() as ctx, staging(out, ".multigrid-") as td:
        stage = Path(td) / "ready"
        stage.mkdir()
        wd = Path(td).resolve() / "scratch"
        wd.mkdir(mode=0o700)

        def hold(p, h, limit, ex=False):
            return ctx.enter_context(snapshot(p, h, limit=limit, executable=ex))

        def proc(fd):
            return f"/proc/self/fd/{fd}"

        geo = hold(ref / "full-corrected.geom", d["geometry_sha256"], 4096)
        pr = hold(ref / "full-corrected.f32", d["projections_sha256"], 256 * MIB)
        ptx = hold(base / "kernels.ptx", PTX, 4 * MIB)
        origexe = hold(base / "cone-fdk", BASE, 8 * MIB, True)
        newexe = hold(bld / "multigrid", br["builds"][0]["multigrid"], 8 * MIB, True)
        vol = hold(ref / "full-corrected-512.f32", d["volume_sha256"], 512 * MIB)
        known = crop(vol, 512, GRIDS[1][2])
        specpath = wd / "grids.txt"
        specpath.write_bytes(spec(GRIDS))
        gf = hold(specpath, hashlib.sha256(spec(GRIDS)).hexdigest(), 4096)
        common = [
            "--geom",
            proc(geo),
            "--input",
            proc(pr),
            "--ptx",
            proc(ptx),
            "--budget",
            str(cap),
            "--batch",
            str(batch),
        ]
        fds = (geo, pr, ptx)
        hashes = []
        packpin = None

        def trial(j, arm, keep=False):
            nonlocal packpin
            natives = []
            times = []
            parts = []
            if arm == "serial":
                for idx, (n, fov, box) in enumerate(GRIDS):
                    outpath = wd / f"part-{idx}"
                    need(not os.path.lexists(outpath), "private output already exists")
                    cmd = [
                        proc(origexe),
                        *common,
                        "--n",
                        str(n),
                        "--fov",
                        str(fov),
                        "--output",
                        outpath,
                    ]
                    full = tuple(box) == (0, 0, 0, n, n, n)
                    if not full:
                        cmd += ["--roi", ",".join(map(str, box))]
                    raw, ms = execute(cmd, (*fds, origexe))
                    guard(raw, scan, cap, batch, [GRIDS[idx]], False)
                    need(outpath.stat().st_size == raw["output_bytes"], "partial payload")
                    key = sha(outpath)
                    parts.append(key)
                    need(not hashes or hashes[idx] == key, "serial did not repeat")
                    need(idx != 1 or known == key, "baseline crop disagrees")
                    natives.append(raw)
                    times.append(ms)
                    if not hashes:
                        continue
                    outpath.unlink()
                if not hashes:
                    hashes.extend(parts)
            else:
                outpath = wd / "volume.mg"
                need(not os.path.lexists(outpath), "private output already exists")
                raw, ms = execute(
                    [proc(newexe), *common, "--grids", proc(gf), "--output", outpath],
                    (*fds, gf, newexe),
                )
                guard(raw, scan, cap, batch, GRIDS, True)
                with outpath.open("rb") as f:
                    meta = review.read(f)
                parts = [v["sha256"] for v in meta["regions"]]
                need(parts == hashes, "one or more independent grids disagree")
                need(packpin is None or packpin == meta["pack_sha256"], "pack nondeterminism")
                packpin = meta["pack_sha256"]
                natives = [raw]
                times = [ms]
                if keep:
                    outpath.rename(stage / "volume.mg")
                else:
                    outpath.unlink()
                if j == -1:
                    for i, key in enumerate(parts):
                        p = wd / f"part-{i}"
                        need(sha(p) == key, "retained comparison changed")
                        p.unlink()
            row = dict(
                block=j,
                arm=arm,
                native=natives,
                process_times_ms=times,
                native_process_ms=sum(times),
                per_grid_sha256=parts,
                scan_bytes=sum(r["projection_bytes_read"] for r in natives),
                peak_owned_bytes=max(r["peak_owned_bytes"] for r in natives),
            )
            print(j, arm, round(sum(times), 3), flush=True)
            return row

        warm = [trial(-1, "serial"), trial(-1, "shared")]
        rows = [
            trial(j, k, keep=j == 5 and k == "shared")
            for j in range(6)
            for k in (("serial", "shared") if j % 2 == 0 else ("shared", "serial"))
        ]
        v = compare(rows)
        meta = review.verify(stage / "volume.mg", packpin)
        doc = dict(
            format="CFDK-GB10-multigrid-qualification-v1",
            parents=dict(plan=PLAN, build=BUILD, original=NATIVE, astra=ASTRA),
            source={p.name: sha(p) for p in [Path(__file__), Path(review.__file__)]},
            geometry=d["geometry_sha256"],
            projection=d["projections_sha256"],
            grid_spec_sha256=hashlib.sha256(spec(GRIDS)).hexdigest(),
            grids=GRIDS,
            budget_mib=cap,
            batch=batch,
            overview_data="direct, not resampled",
            reference_512_crop_sha256=known,
            pack=meta,
            warmups=warm,
            samples=rows,
            decision=v,
            scope="Three independent direct FDK requests of the same 720-view walnut, same batch, one whole 192 grid and two ROIs of 512/1024 parent grids with different FOVs. Native process times include spawn, GPU work and output fsync (3 files vs 1 pack), exclude hash checks, input sealing, and NRRD export. Warm page cache. No spatial resolution or physical dimension accuracy claim.",
        )
        (stage / "qualification.json").write_text(json.dumps(doc, indent=2) + "\n")
        publish(stage, out)
        print(json.dumps(v), sha(out / "qualification.json"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    run(p.parse_args().out)
