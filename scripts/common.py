"""Shared paths, raw format, native CLI and independent ASTRA runner."""

import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "reports"
BUILD = ROOT / "build"


@dataclasses.dataclass(frozen=True)
class Geometry:
    a: int
    v: int
    u: int
    du: float
    dv: float
    R: float
    D: float
    start: float = 0.0
    sign: float = 1.0

    def save(self, p):
        p.write_text("CFDK1 " + " ".join(str(x) for x in dataclasses.astuple(self)) + "\n")

    @classmethod
    def read(cls, p):
        magic, a, v, u, *dims = p.read_text().split()
        assert magic == "CFDK1"
        return cls(int(a), int(v), int(u), *map(float, dims))

    def angles(self):
        return self.start + self.sign * 2 * np.pi * np.arange(self.a) / self.a


def dll_dirs():
    base = pathlib.Path(sys.prefix) / "Lib/site-packages"
    return list((base / "nvidia").glob("*/bin")) + [base / "astra"]


def reconstruct(
    scan,
    out,
    n,
    fov,
    budget=768,
    batch=16,
    *,
    build=BUILD,
    host_memory=None,
    roi=None,
    roi_list=None,
    planner=None,
    plane=None,
    stack=None,
    stack_format=None,
):
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(map(str, dll_dirs())) + os.pathsep + env.get("PATH", "")
    cmd = [
        build / "cone-fdk.exe",
        "--ptx",
        build / "kernels.ptx",
        "--geom",
        scan.with_suffix(".geom"),
        "--input",
        scan.with_suffix(".f32"),
        "--output",
        out,
        "--n",
        n,
        "--fov",
        fov,
        "--budget",
        budget,
        "--batch",
        batch,
    ]
    if roi is not None:
        cmd.extend(["--roi", roi])
    if roi_list is not None:
        cmd.extend(["--roi-list", roi_list])
    if plane is not None:
        cmd.extend(["--plane", plane])
    if stack is not None:
        cmd.extend(["--stack", stack])
    if stack_format is not None:
        cmd.extend(["--stack-format", stack_format])
    if planner is not None:
        cmd.extend(["--planner", planner])
    if host_memory is not None:
        cmd.extend(["--host-memory", host_memory])
    t = time.perf_counter()
    r = subprocess.run(
        list(map(str, cmd)),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - t
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    d = json.loads(r.stdout)
    d["process_seconds"] = elapsed
    return d


def astra_run(scan, out, n, fov, *, centered_shape_xyz=None):
    # Same physical mm, unmodified absolute scale. No post-hoc registration or gain fit.
    _dll_directory_cookies = [os.add_dll_directory(str(p)) for p in dll_dirs() if p.is_dir()]
    import astra

    g = Geometry.read(scan.with_suffix(".geom"))
    t = time.perf_counter()
    s = np.memmap(scan.with_suffix(".f32"), "float32", "r", shape=(g.a, g.v, g.u))
    pg = astra.create_proj_geom("cone", g.du, g.dv, g.v, g.u, g.angles(), g.R, g.D - g.R)
    # For independent ROI verification use a centered box enclosing the requested
    # patch. No detector cropping or translation of ASTRA's rotation axis.
    nx, ny, nz = centered_shape_xyz or (n, n, n)
    if any(not 1 <= k <= n or (n - k) % 2 for k in (nx, ny, nz)):
        raise ValueError("centered reference must align with the parent grid")
    rx, ry, rz = (k * (fov / n) / 2 for k in (nx, ny, nz))
    vg = astra.create_vol_geom(ny, nx, nz, -rx, rx, -ry, ry, -rz, rz)
    ids = []
    alg = None
    try:
        ids.append(astra.data3d.create("-sino", pg, np.ascontiguousarray(s.transpose(1, 0, 2))))
        ids.append(astra.data3d.create("-vol", vg))
        cfg = astra.astra_dict("FDK_CUDA")
        cfg.update(
            ProjectionDataId=ids[0],
            ReconstructionDataId=ids[1],
            option={"FilterType": "ram-lak"},
        )
        alg = astra.algorithm.create(cfg)
        t1 = time.perf_counter()
        astra.algorithm.run(alg)
        run = time.perf_counter() - t1
        vol = astra.data3d.get_shared(ids[1])
        vol.tofile(out)
        del vol
    finally:
        if alg is not None:
            astra.algorithm.delete(alg)
        for i in ids:
            astra.data3d.delete(i)
    return {
        "astra_version": astra.__version__,
        "algorithm_seconds": run,
        "io_algorithm_seconds": time.perf_counter() - t,
    }


def difference(lhs, rhs, n):
    x = np.memmap(lhs, "f4", "r", shape=(n, n, n))
    y = np.memmap(rhs, "f4", "r", shape=x.shape)
    sse = ref = 0.0
    mx = 0.0
    peak = 0.0
    for i in range(n):
        a = x[i].astype("f8")
        b = y[i].astype("f8")
        delta = a - b
        sse += (delta * delta).sum()
        ref += (b * b).sum()
        mx = max(mx, float(np.abs(delta).max()))
        peak = max(peak, float(np.abs(b).max()))
    return {
        "rel_l2": float(np.sqrt(sse / max(ref, 1e-30))),
        "max_abs": mx,
        "ref_peak": peak,
    }
