"""Replay two ROI gates on the original 720-view walnut projections. No full 2048³ output."""

import argparse
import json
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, DATA, REPORT, astra_run, reconstruct
from compare_native import sha
from PIL import Image, ImageDraw


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def region(box, origin=(0, 0, 0)):
    return tuple(slice(a - b, a - b + k) for a, b, k in zip(box[:3], origin, box[3:]))[::-1]


def stats(out, shape, ref):
    a = np.fromfile(out, "<f4").reshape(shape).astype("f8")
    b = ref.astype("f8")
    d = a - b
    return dict(
        relative_l2=float(np.sqrt((d * d).sum() / (b * b).sum())), max_abs=float(np.abs(d).max())
    )


def overview(full, direct, box, path):
    x, y, z, w, h, d = box
    layer = full[z + d // 2]
    lo, hi = np.percentile(layer, [2, 99])

    def window(frame):
        return Image.fromarray(
            (np.clip((frame - lo) / (hi - lo), 0, 1) * 255).astype("u1")
        ).convert("RGB")

    canvas = Image.new("RGB", (816, 340), "#192126")
    left = window(layer).resize((280, 280), Image.Resampling.BILINEAR)
    mark = ImageDraw.Draw(left)
    s = 280 / layer.shape[0]
    mark.rectangle((x * s, y * s, (x + w) * s - 1, (y + h) * s - 1), outline="#ffca68", width=2)
    roi = np.fromfile(direct, "<f4").reshape(d, h, w)
    canvas.paste(left, (16, 36))
    canvas.paste(window(roi[d // 2]).resize((280, 280), Image.Resampling.NEAREST), (326, 36))
    ink = ImageDraw.Draw(canvas)
    ink.text((16, 12), f"Parent / 512^3 / z = {z + d // 2}", fill="white")
    ink.text((326, 12), "Direct CUDA ROI / same display window", fill="white")
    ink.multiline_text(
        (622, 88),
        f"{w} x {h} x {d} voxels\n\nX start  {x}\nY start  {y}\nZ start  {z}\n\n0.078125 mm / voxel",
        fill="#f6d69b",
        spacing=5,
    )
    ink.text(
        (16, 320),
        "Helsinki walnut / A. Meaney / CC BY 4.0. ROI enlarged for display; not higher acquisition resolution.",
        fill="#bec8c9",
    )
    canvas.save(path)


def run(scan, reports, scratch):
    reports.mkdir(parents=True, exist_ok=True)
    released_sha = json.loads((REPORT / "run-repeat.json").read_text())["baseline_outputs"][
        "walnut512"
    ]
    baseline = json.loads((REPORT / "native-ab.json").read_text())["cases"]["full-corrected"]
    hashes = {ext: sha(scan.with_suffix(ext)) for ext in (".geom", ".f32")}
    require(
        hashes[".geom"] == baseline["geom_sha256"]
        and hashes[".f32"] == baseline["projections_sha256"],
        "expected the pinned 720-view binned/center-corrected walnut scan",
    )
    outdata = dict(
        dataset="Helsinki walnut, 720 views, 8x8 detector binning; CC BY 4.0; see NOTICE.md",
        input_sha256=hashes,
        binaries_sha256={name: sha(BUILD / name) for name in ("cone-fdk.exe", "kernels.ptx")},
    )
    with TemporaryDirectory(dir=scratch) as tmp:
        tmp = Path(tmp)
        crop = tmp / "window.f32"
        full = tmp / "full.f32"
        box = (289, 161, 197, 128, 128, 96)

        def job(n, box, path, budget):
            m = reconstruct(
                scan, path, n, 40, budget, 16, roi=",".join(map(str, box)) if box else None
            )
            require(
                m["redzones_verified"]
                and m["peak_owned_bytes"] <= budget * 2**20
                and m["output_bytes"] == path.stat().st_size,
                "buffer/length gate failed",
            )
            return m

        fullruns = []
        roiruns = []
        # Warm each mode once, then alternate order. Hashes and Python checks are outside timing.
        job(512, None, full, 768)
        job(512, box, crop, 768)
        pair_hashes = []
        for i in range(6):
            for is_roi in [False, True] if i % 2 == 0 else [True, False]:
                (roiruns if is_roi else fullruns).append(
                    job(512, box if is_roi else None, crop if is_roi else full, 768)
                )
            require(
                sha(full) == released_sha,
                "full-grid output differs from released reference",
            )
            parent = np.memmap(full, "<f4", "r", shape=(512,) * 3)
            require(
                crop.read_bytes() == parent[region(box)].tobytes(),
                "512-grid ROI differs from crop",
            )
            if i == 5:
                overview(parent, crop, box, reports / "roi-walnut.png")
            del parent
            pair_hashes.append(sha(crop))
        require(len(set(pair_hashes)) == 1, "nonrepeatable ROI output")
        med = [
            statistics.median(m["process_seconds"] for m in group)
            for group in [fullruns, roiruns]
        ]
        outdata["same_grid_512"] = dict(
            protocol="One untimed warm-up each, six warm AB/BA pairs on the same GPU. End-to-end subprocess time includes input validation, GPU, flush and output. No clock locking or OS cache eviction.",
            full=fullruns,
            roi=roiruns,
            full_median_seconds=med[0],
            roi_median_seconds=med[1],
            speed_ratio=med[0] / med[1],
            full_sha256=released_sha,
            roi_sha256=pair_hashes[0],
            exact_pairs=len(pair_hashes),
        )
        # ASTRA center stays at the scan isocenter. Solve a larger enclosing box,
        # then extract an off-axis patch from it independently of the CUDA ROI.
        n = 2048
        box = (989, 1105, 975, 161, 113, 79)
        enclosing = tuple(2 * max(n // 2 - a, a + k - n // 2) for a, k in zip(box[:3], box[3:]))
        origin = tuple((n - k) // 2 for k in enclosing)
        m = job(n, box, crop, 128)
        astra_path = tmp / "reference.f32"
        refstats = astra_run(scan, astra_path, n, 40, centered_shape_xyz=enclosing)
        buf = np.memmap(astra_path, "<f4", "r", shape=enclosing[::-1])
        diff = stats(crop, box[3:][::-1], buf[region(box, origin)])
        del buf
        require(diff["relative_l2"] < 0.001, "2048-grid independent ROI relative L2 exceeds 0.1%")
        outdata["fine_grid_2048"] = dict(
            native=m,
            roi_sha256=sha(crop),
            full_grid_bytes_if_written=4 * n**3,
            astra=refstats,
            astra_enclosing_shape_xyz=enclosing,
            astra_enclosing_origin_xyz=origin,
            relative_l2_limit=0.001,
            **diff,
        )
    (reports / "roi-walnut.json").write_text(
        json.dumps(outdata, indent=2) + "\n", encoding="utf8"
    )
    print(
        f"512 grid: {med[0]:.4f} -> {med[1]:.4f} s, 6/6 exact; 2048 grid: {m['output_bytes']} bytes, rel L2 {diff['relative_l2']:.8f}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=DATA / "full-corrected")
    p.add_argument("--reports", type=Path, default=REPORT)
    p.add_argument("--scratch", type=Path, default=None)
    a = p.parse_args()
    run(a.scan, a.reports, a.scratch)
