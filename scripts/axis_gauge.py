"""Locate a straight round tube and regauge perpendicular sections from the original scan."""

import argparse
import ctypes
import json
import shutil
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from pathlib import Path

from bore import gauge
from common import BUILD, reconstruct
from compare_native import sha
from wall import measure, run_native_json


def fit_axis(wall, grid, out, limit, *, budget=16, build=BUILD):
    return run_native_json(
        [
            build / "cone-fdk.exe",
            "axis",
            "--ptx",
            build / "kernels.ptx",
            "--input",
            wall,
            "--cylinder",
            grid,
            "--output",
            out,
            "--max-residual-mm",
            limit,
            "--budget",
            budget,
        ]
    )


@contextmanager
def pinned_reader(path):
    """Deny writers and replacement while separate native processes reopen the same scan."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    open_file = kernel.CreateFileW
    open_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    open_file.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handle = open_file(str(path.resolve()), 0x80000000, 1, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        close(handle)


def run(scan, grid, job, *, fov, threshold, axis_limit, bore_limit, budget=96, batch=16):
    """Fresh job only. The manifest is the commit record, written while inputs are still locked read-only."""
    scan, grid, job = Path(scan), Path(grid), Path(job)
    sources = {k: scan.with_suffix(k) for k in (".geom", ".f32")}
    sources.update({n: BUILD / n for n in ("cone-fdk.exe", "kernels.ptx")})
    with ExitStack() as held:
        for k in (".geom", ".f32", "kernels.ptx", "cone-fdk.exe"):
            held.enter_context(pinned_reader(sources[k]))
        return _run_locked(
            scan, grid, job, sources, fov, threshold, axis_limit, bore_limit, budget, batch
        )


def _run_locked(scan, grid, job, sources, fov, threshold, axis_limit, bore_limit, budget, batch):
    def fingerprints():
        return {k: sha(v) for k, v in sources.items()}

    job.mkdir(parents=True, exist_ok=False)
    inputs = fingerprints()
    coarse = job / "coarse.grid"
    shutil.copyfile(grid, coarse)
    log = {
        "format": "CFDK-axis-job-v1",
        "inputs_sha256": inputs,
        "threshold": threshold,
        "fov_mm": fov,
        "axis_limit_mm": axis_limit,
        "bore_limit_mm": bore_limit,
        "budget_mib": budget,
        "batch": batch,
        "stages": {},
    }

    def stage(frame):
        spec, atlas, wall = (job / (frame + ext) for ext in (".grid", ".atlas", ".wall"))
        rec = reconstruct(scan, atlas, 512, fov, budget, batch, cylinder=spec)
        stats = measure(atlas, spec, wall, threshold, budget=budget)
        if not (
            rec["redzones_verified"]
            and stats["redzones_verified"]
            and rec["peak_owned_bytes"] <= rec["effective_budget_bytes"] <= rec["budget_bytes"]
            and stats["peak_owned_bytes"] <= stats["budget_bytes"]
            and rec["output_bytes"] == atlas.stat().st_size == stats["input_bytes"]
            and stats["output_bytes"] == wall.stat().st_size
            and rec["projection_bytes_read"]
            == rec["projection_passes"] * sources[".f32"].stat().st_size
        ):
            raise RuntimeError("stage accounting mismatch: " + frame)
        log["stages"][frame] = {
            "projection_passes": rec["projection_passes"],
            "projection_bytes_read": rec["projection_bytes_read"],
            "reconstruction_peak_owned_bytes": rec["peak_owned_bytes"],
            "wall": stats,
        }
        return stats

    meta = stage("coarse")
    axis_path = job / "axis.json"
    info = fit_axis(job / "coarse.wall", coarse, axis_path, axis_limit, budget=budget)
    log["stages"]["axis"] = info
    fit = json.loads(axis_path.read_bytes())
    m = fit["measurement"]
    status = "axis_rejected"
    if m is not None:
        h, w = meta["shape_ha"]
        fields = [
            (w, h, meta["shells"]),
            m["origin_xyz_mm"],
            m["axis_unit"],
            m["seam_unit"],
            [meta[k] for k in ("first_radius_mm", "radial_step_mm", "axial_pitch_mm")],
        ]
        aligned = job / "aligned.grid"
        aligned.write_text(
            "CFDKCYL1 " + "\n".join(" ".join(map(str, line)) for line in fields) + "\n",
            encoding="ascii",
            newline="\n",
        )
        stage("aligned")
        g = gauge(job / "aligned.wall", aligned, job / "sections.json", bore_limit, budget=budget)
        log["stages"]["bore"] = g
        status = "measured" if g["measured"] == h else "sections_rejected"
    log["status"] = status
    log["axis_status"] = fit["status"]
    log["outputs_sha256"] = {p.name: sha(p) for p in sorted(job.iterdir())}
    tmp = job / "manifest.partial.json"
    tmp.write_text(
        json.dumps(log, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
    )
    tmp.replace(job / "manifest.json")
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("scan", "grid", "job"):
        p.add_argument(key, type=Path)
    for key in ("fov", "threshold", "axis-limit", "bore-limit"):
        p.add_argument("--" + key, type=float, required=True)
    p.add_argument("--budget", type=int, default=96)
    p.add_argument("--batch", type=int, default=16)
    args = vars(p.parse_args())
    result = run(**args)
    print(
        json.dumps({"status": result["status"], "manifest": str(args["job"] / "manifest.json")})
    )
    raise SystemExit(0 if result["status"] == "measured" else 2)
