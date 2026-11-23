"""Reconstruct >8 GiB with 512 MiB of owned allocations; verify five physical Z planes
independently with ASTRA, without constructing a second 8-GiB volume. Temp output is removed.
"""

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile

from common import DATA, REPORT, Geometry, dll_dirs, np, reconstruct


def run():
    n = 1296
    fov = 40.5
    step = fov / n
    scan = DATA / "phantom"
    g = Geometry.read(scan.with_suffix(".geom"))
    _dll_directory_cookies = [os.add_dll_directory(str(p)) for p in dll_dirs() if p.is_dir()]
    import astra

    total = (
        int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        * 1024**2
    )
    with tempfile.TemporaryDirectory(dir=DATA) as tmp:
        out = pathlib.Path(tmp) / "oversize.raw"
        stats = reconstruct(scan, out, n, fov, 512, 16)
        size = out.stat().st_size
        assert size == n**3 * 4 and size > total
        assert stats["slabs"] > 1 and stats["peak_owned_bytes"] <= 512 * 1024**2
        mm = np.memmap(out, "f4", "r", shape=(n, n, n))
        pg = astra.create_proj_geom("cone", g.du, g.dv, g.v, g.u, g.angles(), g.R, g.D - g.R)
        views = np.memmap(scan.with_suffix(".f32"), "f4", "r", shape=(g.a, g.v, g.u))
        pid = astra.data3d.create("-sino", pg, np.ascontiguousarray(views.transpose(1, 0, 2)))
        checks = []
        try:
            # Probe both sides of a physical slab seam, center, and interior +/- offsets.
            seam = stats["slab"] * (n // (2 * stats["slab"]))
            for z in sorted(set([n // 3, seam - 1, seam, n // 2, 2 * n // 3])):
                zz = -fov / 2 + z * step
                vg = astra.create_vol_geom(
                    n, n, 1, -fov / 2, fov / 2, -fov / 2, fov / 2, zz, zz + step
                )
                vid = astra.data3d.create("-vol", vg)
                cfg = astra.astra_dict("FDK_CUDA")
                cfg.update(ProjectionDataId=pid, ReconstructionDataId=vid)
                aid = astra.algorithm.create(cfg)
                try:
                    astra.algorithm.run(aid)
                    ref = astra.data3d.get(vid)[0].astype("f8")
                    rel = float(np.linalg.norm(mm[z].astype("f8") - ref) / np.linalg.norm(ref))
                    assert rel < 0.01, (z, rel)
                    checks.append({"z": z, "relative_l2": rel})
                finally:
                    astra.algorithm.delete(aid)
                    astra.data3d.delete(vid)
        finally:
            astra.data3d.delete(pid)
            del mm
        with out.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        result = dict(
            native=stats,
            output_bytes=size,
            gpu_total_bytes_smi=total,
            sha256=sha,
            checked_planes=checks,
            verification="Five physical planes only; not a full ASTRA volume.",
        )
        (REPORT / "oversize.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
