"""Independent ASTRA CUDA witnesses for the three measured, directly reconstructed walnut grids."""

import argparse, hashlib, json, math, os, struct, subprocess, time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from grouped_replay import M, SCAN, need, sha, json_checked, ASTRA
from qualify_multigrid import GRIDS
from review_multigrid import verify, fp32
from sealed_inputs import snapshot
from spark_publication import fresh, staging, publish

PROFILE = "73eb45ab5c65165132b7038e5d196289373d90eb9892ec6cc6157bff8eaaeee8"
DELIVERY = "bdd4278e296c53afbae8bcb405abfaa64ab8d158bbf89d7b4d345217e6c336db"
PACK = "806aaf2511308d5ebf105921452bfbc05827c6db261be2e1b0266e6d62b1f530"
LIB = "f2a432dd81f2046c211c9f24a8ad92cec5587cb533bed640811a7317ceff2250"
LIMIT = (0.002, 0.0002)
CONTROL = (0.0001, 0.00002)


def enclosure(n, box):
    need(
        type(n) is int and n > 0 and len(box) == 6 and all(type(v) is int for v in box),
        "integer Cartesian grid required",
    )
    o, s = box[:3], box[3:]
    need(all(0 <= i < i + l <= n for i, l in zip(o, s, strict=True)), "invalid interval")
    m = [min(i, n - i - l) for i, l in zip(o, s, strict=True)]
    return [n - 2 * k for k in m], m, [i - k for i, k in zip(o, m, strict=True)]


def roi(v, o, s):
    return v[tuple(slice(i, i + l) for i, l in zip(reversed(o), reversed(s), strict=True))]


def score(x, y, origin=(0, 0, 0)):
    need(
        x.shape == y.shape and x.ndim == 3 and min(x.shape) > 0,
        "reference must cover every voxel",
    )
    planes = []
    hi = -1.0
    loc = None
    for z, (a, b) in enumerate(zip(x, y, strict=True)):
        x64 = a.astype(np.float64)
        r = b.astype(np.float64)
        d = x64 - r
        need(np.isfinite(d).all() and np.isfinite(r).all(), "nonfinite volume")
        mag = np.abs(d)
        ix = int(np.argmax(mag))
        val = float(mag.flat[ix])
        j, i = divmod(ix, a.shape[1])
        if val > hi:
            hi = val
            loc = [origin[0] + i, origin[1] + j, origin[2] + z]
        planes.append(
            dict(
                z=origin[2] + z,
                sse=float(np.sum(d * d)),
                energy=float(np.sum(r * r)),
                signed_sum=float(np.sum(d)),
                max_abs=val,
            )
        )
    e = math.fsum(p["energy"] for p in planes)
    need(e > 0, "blank reference")
    return dict(
        voxels=int(x.size),
        rel_l2=math.sqrt(math.fsum(p["sse"] for p in planes) / e),
        signed_mean=math.fsum(p["signed_sum"] for p in planes) / x.size,
        max_abs=hi,
        max_parent_xyz=loc,
        planes=planes,
    )


def passes(row, limits):
    return row["rel_l2"] <= limits[0] and row["max_abs"] <= limits[1]


def geometry(raw):
    t = raw.split()
    need(len(t) == 10 and t[0] == b"CFDK1", "geometry header")
    a, v, u = map(int, t[1:4])
    du, dv, R, D, start, sign = map(float, t[4:])
    need(
        min(a, v, u) > 0
        and all(math.isfinite(k) for k in (du, dv, R, D, start, sign))
        and min(du, dv, R) > 0
        and D > R
        and sign in (1, -1),
        "geometry bounds",
    )
    return (a, v, u), (du, dv, R, D, start, sign)


def proj(astra, dims, params):
    a, v, u = dims
    du, dv, R, D, start, sign = params
    return astra.create_proj_geom(
        "cone", du, dv, v, u, start + sign * 2 * np.pi * np.arange(a) / a, R, D - R
    )


def reconstruct(astra, pid, n, fov, box):
    s, m, o = enclosure(n, box)
    rx, ry, rz = [v * (fov / n) / 2 for v in s]
    vg = astra.create_vol_geom(s[1], s[0], s[2], -rx, rx, -ry, ry, -rz, rz)
    vid = astra.data3d.create("-vol", vg)
    alg = None
    try:
        cfg = astra.astra_dict("FDK_CUDA")
        cfg.update(
            ProjectionDataId=pid,
            ReconstructionDataId=vid,
            option={"FilterType": "ram-lak", "GPUindex": 0},
        )
        alg = astra.algorithm.create(cfg)
        t = time.perf_counter()
        astra.algorithm.run(alg)
        elapsed = time.perf_counter() - t
        # Own the crop before the ASTRA-managed backing volume is released.
        part = np.array(
            roi(astra.data3d.get_shared(vid), o, box[3:]), dtype="<f4", order="C", copy=True
        )
    finally:
        if alg is not None:
            astra.algorithm.delete(alg)
        astra.data3d.delete(vid)
    return part, dict(
        enclosing_shape_xyz=s, enclosure_origin_xyz=m, local_start_xyz=o, algorithm_s=elapsed
    )


def libcheck(astra, base):
    device = astra.get_gpu_info(0)
    paths = {
        t.split()[-1]
        for t in Path("/proc/self/maps").read_text().splitlines()
        if "/libastra.so." in t
    }
    lib = (M / "vendor/astra-install/lib/libastra.so.0").resolve()
    need(paths == {str(lib)} and sha(lib) == LIB, "loaded ASTRA library")
    commit = subprocess.check_output(
        ["git", "-C", M / "vendor/astra-2.5.0", "rev-parse", "HEAD"], text=True
    ).strip()
    need(
        astra.__version__ == base["astra"]["astra_version"]
        and device == base["device"]
        and commit == base["astra_source_commit"],
        "ASTRA version/device/source",
    )
    return dict(
        device=device, version=astra.__version__, source_commit=commit, loaded_library_sha256=LIB
    )


def run(out):
    import astra

    out = fresh(out)
    prof = json_checked(M / "multigrid-qualification-001/qualification.json", PROFILE)
    delivery = json_checked(M / "multigrid-delivery-001/delivery.json", DELIVERY)
    parent = json_checked(M / "walnut-astra-gb10-001/result.json", ASTRA)
    need(
        parent["passed"] is True
        and delivery["profile"] == PROFILE
        and prof["pack"]["pack_sha256"] == PACK,
        "parents",
    )
    env = libcheck(astra, parent)
    pins = parent["input_sha256"]
    inputs = {}
    rows = []
    tests = []
    with ExitStack() as stack, staging(out, ".grid-oracle-") as td:
        stage = Path(td) / "ready"
        stage.mkdir()

        def sealed(path, pin, size):
            fd = stack.enter_context(snapshot(path, pin, exact=size, limit=size))
            inputs[path.name] = pin
            return Path(f"/proc/self/fd/{fd}")

        # Pin both inputs and every volume actually mapped; never hash and then reopen a mutable source.
        geom = sealed(SCAN.with_suffix(".geom"), pins["full-corrected.geom"], 49)
        dims, params = geometry(geom.read_bytes())
        sg = sealed(SCAN.with_suffix(".f32"), pins["full-corrected.f32"], 4 * math.prod(dims))
        pack = sealed(M / "multigrid-delivery-001/volume.mg", PACK, 50474220)
        nr = verify(pack, PACK)
        need(
            nr == prof["pack"]
            and prof["geometry"] == pins["full-corrected.geom"]
            and prof["projection"] == pins["full-corrected.f32"],
            "scan/grid identity",
        )
        prior = sealed(
            M / "walnut-astra-gb10-001/astra-gb10-512.f32", parent["sha256"], 4 * 512**3
        )
        full = np.memmap(prior, dtype="<f4", mode="r", shape=(512,) * 3)
        pg = proj(astra, dims, params)
        scan = np.memmap(sg, dtype="<f4", mode="r", shape=dims)
        pid = astra.data3d.create("-sino", pg, np.ascontiguousarray(scan.transpose(1, 0, 2)))
        stack.callback(astra.data3d.delete, pid)
        refpack = stage / "astra.mg"
        with refpack.open("xb") as dest:
            dest.write(struct.pack("<8sI", b"CFDKMG01", len(GRIDS)))
            for n, fov, box in GRIDS:
                dest.write(struct.pack("<Iff6I", n, fov, fp32(fov / n), *box))
            for reg, (n, fov, box) in zip(nr["regions"], GRIDS, strict=True):
                need(
                    (reg["n"], reg["fov_mm"], [*reg["origin_xyz"], *reg["shape_xyz"]])
                    == (n, fov, list(box)),
                    "independent grid request disagrees with native pack",
                )
                x = np.memmap(
                    pack,
                    dtype="<f4",
                    mode="r",
                    offset=reg["offset_bytes"],
                    shape=tuple(reversed(box[3:])),
                )
                y, w = reconstruct(astra, pid, n, fov, box)
                row = score(x, y, box[:3])
                assertion = passes(row, LIMIT)
                need(
                    assertion,
                    "fresh ASTRA numerical gate: " + str((n, row["rel_l2"], row["max_abs"])),
                )
                y.tofile(dest)
                entry = dict(
                    parent_n=n, fov_mm=fov, box=list(box), witness=w, native_vs_astra=row
                )
                if n == 512:
                    old = roi(full, box[:3], box[3:])
                    control = score(y, old, box[:3])
                    need(passes(control, CONTROL), "prior full ASTRA crop disagrees")
                    entry["fresh_vs_prior_full_astra"] = control
                    for axis in range(3):
                        off = list(box[:3])
                        off[axis] += 1
                        wrong = score(x, roi(full, off, box[3:]), box[:3])
                        need(not passes(wrong, LIMIT), "missed shifted-origin negative")
                        tests.append(
                            dict(
                                shift_one_voxel_xyz=axis,
                                rel_l2=wrong["rel_l2"],
                                max_abs=wrong["max_abs"],
                            )
                        )
                rows.append(entry)
        rp = sha(refpack)
        rd = verify(refpack, rp)
        doc = dict(
            format="CFDK-Spark-independent-grid-oracle-v1",
            parents=dict(multigrid=PROFILE, delivery=DELIVERY, astra=ASTRA),
            inputs=inputs,
            astra=env,
            source_sha256=sha(Path(__file__)),
            native_pack_sha256=PACK,
            oracle_pack=rd,
            limits=dict(
                native_rel_l2=LIMIT[0],
                native_max_abs=LIMIT[1],
                control_rel_l2=CONTROL[0],
                control_max_abs=CONTROL[1],
            ),
            regions=rows,
            negative_shifts=tests,
            scope="one measured walnut; no gain fitting, registration, interpolation or physical dimensional accuracy claim",
            passed=True,
        )
        (stage / "qualification.json").write_text(
            json.dumps(doc, indent=2, allow_nan=False) + "\n"
        )
        publish(stage, out)
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    d = run(a.out)
    print(
        json.dumps(
            [
                (r["parent_n"], r["native_vs_astra"]["rel_l2"], r["native_vs_astra"]["max_abs"])
                for r in d["regions"]
            ]
        )
    )
