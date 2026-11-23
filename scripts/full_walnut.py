"""All 720 unique angles of Meaney's measured walnut, with bounded host memory.
Never keeps more than a few original TIFFs per worker; exact equality against the
90-view offline fixture is checked before using either full scan for reconstruction.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import time
import zipfile

from accept import gated, png_grid
from common import DATA, REPORT, ROOT, Geometry, astra_run, difference, np, reconstruct
from export_volume import export
from fetch import MD5, SIZES, digest
from prepare_walnut import bin_views


def full(jobs=3, wait_seconds=0):
    DATA.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)
    g = Geometry(720, 296, 280, 0.4, 0.4, 210.66, 553.74)
    scans = [DATA / ("full-" + s) for s in ("uncorrected", "corrected")]
    fixture = ROOT / "fixtures" / "walnut-s8-b8"
    pinned = json.loads((fixture / "frames.json").read_text())
    manifests = {}
    md5 = {}
    shape = (g.a, g.v, g.u)
    # Temporary inputs; do not publish until verified against the independent packaged subset.
    work = [p.with_suffix(".f32.part") for p in scans]
    mmap = [np.memmap(p, dtype="<f4", mode="w+", shape=shape) for p in work]
    for i in range(1, 9):
        path = DATA / "downloads" / f"20201111_walnut_projections_{i}.zip"
        deadline = time.monotonic() + wait_seconds
        while not path.exists() or path.stat().st_size != SIZES[i - 1]:
            if time.monotonic() >= deadline:
                raise FileNotFoundError(f"Run scripts/fetch.py first: {path}")
            time.sleep(3)
        md5[path.name] = digest(path)
        gated(md5[path.name] == MD5[i - 1], f"archive {i} failed checksum")
        with (
            zipfile.ZipFile(path) as archive,
            cf.ThreadPoolExecutor(max_workers=jobs) as pool,
        ):
            names = [n for n in archive.namelist() if n.endswith(".tif") and int(n[-8:-4]) <= 720]

            def one(name):
                raw = archive.read(name)  # verifies ZIP member CRC
                return name, hashlib.sha256(raw).hexdigest(), bin_views(raw, 8)

            # Bound pending work and memory even if the first TIFF is slow.
            for start in range(0, len(names), jobs * 2):
                for name, sha, pair in pool.map(one, names[start : start + jobs * 2]):
                    idx = int(name[-8:-4]) - 1
                    gated(
                        0 <= idx < 720 and idx not in manifests,
                        "duplicate/invalid frame",
                    )
                    gated(
                        name not in pinned or pinned[name] == sha,
                        "pinned frame mismatch",
                    )
                    manifests[idx] = (name, sha)
                    for mm, img in zip(mmap, pair):
                        gated(
                            img.shape == (g.v, g.u) and np.isfinite(img).all(),
                            "bad projections",
                        )
                        mm[idx] = img
        print(f"archive {i}/8 processed: {len(manifests)} measured views", flush=True)
    gated(set(manifests) == set(range(720)), "incomplete rotation")
    for mm, p in zip(mmap, scans):
        mm.flush()
        with np.load(fixture / (p.name[5:] + ".npz")) as subset:
            gated(
                np.array_equal(mm[::8], subset["views"]),
                "full/subsample preprocessing mismatch",
            )
    del mm, mmap
    for p, w in zip(scans, work):
        os.replace(w, p.with_suffix(".f32"))
        g.save(p.with_suffix(".geom"))
    (REPORT / "walnut-full-frames.json").write_text(
        json.dumps(dict(manifests[k] for k in sorted(manifests)), indent=2) + "\n"
    )
    n = 512
    fov = 40.0
    outs = [DATA / (p.name + "-512.f32") for p in scans]
    stats = [reconstruct(p, o, n, fov, 768, 16) for p, o in zip(scans, outs)]
    ref = DATA / "full-astra-512.f32"
    base = astra_run(scans[1], ref, n, fov)
    diff = difference(outs[1], ref, n)
    axis_effect = difference(outs[0], outs[1], n)
    gated(
        diff["rel_l2"] < 0.02 and axis_effect["rel_l2"] > 0.001,
        "real-data/rotation-center gate failed",
    )
    gated(
        all(s["peak_owned_bytes"] <= s["budget_bytes"] and s["redzones_verified"] for s in stats),
        "memory/guard gate failed",
    )
    u, c, a = [np.memmap(p, "<f4", "r", shape=(n, n, n)) for p in [*outs, ref]]
    mid = n // 2
    gated(float(np.ptp(c[mid])) > 0.005, "blank volume")
    png_grid(
        [
            ("Before center correction", u[mid]),
            ("Native CUDA / corrected", c[mid]),
            ("ASTRA / corrected", a[mid]),
        ],
        REPORT / "walnut-full.png",
    )
    png_grid(
        [
            ("Axial / XY", c[mid]),
            ("Coronal / XZ", c[:, mid, :]),
            ("Sagittal / YZ", c[:, :, mid]),
        ],
        REPORT / "walnut-full-orthogonal.png",
    )
    export(outs[1], n, fov, REPORT / "full", "Inside a walnut / 720 measured views")
    with outs[1].open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    res = dict(
        views=g.a,
        detector_rows=g.v,
        detector_cols=g.u,
        binning=8,
        n=n,
        fov_mm=fov,
        archives_md5=md5,
        frame_manifest="walnut-full-frames.json",
        duplicate_360_excluded=True,
        offline_subset_bitwise_equal=True,
        native=stats,
        astra=base,
        corrected_vs_astra=diff,
        uncorrected_vs_corrected=axis_effect,
        output_sha256=sha,
    )
    (REPORT / "walnut-full.json").write_text(json.dumps(res, indent=2) + "\n")
    print("Full-volume real scan gate passed:", diff, "output SHA256:", sha, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 17), default=3)
    parser.add_argument(
        "--wait-archives",
        type=int,
        default=0,
        help="maximum seconds to wait for each archive in a concurrent fetch",
    )
    a = parser.parse_args()
    full(a.jobs, a.wait_archives)
