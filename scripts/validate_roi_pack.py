"""Qualify six shared ROIs against serial jobs and the known full walnut volume."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, DATA, REPORT, ROOT, reconstruct
from compare_native import sha
from PIL import Image, ImageDraw, ImageFont
from roi_pack import read_index
from validate_roi import region, require

LIST = ROOT / "fixtures/walnut-six-rois.txt"
KEYS = (
    "process_seconds",
    "projection_passes",
    "filter_batches",
    "projection_bytes_read",
    "launches_bp",
    "peak_owned_bytes",
    "filter_ms",
    "bp_ms",
)


def preview(parent, pack, entries, target):
    """Parent and data actually read from the pack; same display window across all six pairs."""
    lo, hi = np.percentile(parent[::8, ::8, ::8], [2, 99])
    canvas = Image.new("RGB", (1060, 525), "#172228")
    draw = ImageDraw.Draw(canvas)

    def label(xy, text, size=16, color="#e8eeea"):
        draw.text(xy, text, font=ImageFont.load_default(size=size), fill=color)

    def grayscale(frame):
        return Image.fromarray(
            (np.clip((frame - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
        ).convert("RGB")

    label((16, 10), "Six inspection regions from one CT scan", 25)
    label(
        (16, 44),
        "Left: parent slice with selected box.  Right: the packed ROI, enlarged.  All 12 images share one intensity window.",
        15,
    )
    for i, e in enumerate(entries):
        x, y, z = e["origin_xyz"]
        nx, ny, nz = e["shape_xyz"]
        local = nz // 2
        ox, oy = 16 + 346 * (i % 3), 84 + 205 * (i // 3)
        label(
            (ox, oy), f"{i + 1:02d}    z = {z + local}    |    {nx} x {ny} x {nz}", 16, "#f8cb80"
        )
        small = grayscale(parent[z + local]).resize((152, 152), Image.Resampling.BILINEAR)
        scale = 152 / parent.shape[0]
        ImageDraw.Draw(small).rectangle(
            (x * scale, y * scale, (x + nx) * scale - 1, (y + ny) * scale - 1),
            outline="#ffcb65",
            width=2,
        )
        canvas.paste(small, (ox, oy + 26))
        vol = np.memmap(pack, dtype="<f4", mode="r", offset=e["offset_bytes"], shape=(nz, ny, nx))
        patch = grayscale(vol[local])
        del vol
        patch = patch.resize(
            (round(152 * nx / max(nx, ny)), round(152 * ny / max(nx, ny))),
            Image.Resampling.NEAREST,
        )
        canvas.paste(
            patch, (ox + 164 + (152 - patch.width) // 2, oy + 26 + (152 - patch.height) // 2)
        )
    label(
        (16, 495),
        "Data: Alexander Meaney / University of Helsinki, CC BY 4.0.  FOV 40 mm; parent 512 cubed.  Display crops are not defect labels.",
        14,
        "#bccbc8",
    )
    canvas.save(target)


def run(scan, reports, scratch):
    specs = [s for s in LIST.read_text().splitlines() if s and not s.startswith("#")]
    boxes = [list(map(int, s.split(","))) for s in specs]
    require(len(boxes) == 6, "six-box comparison fixture changed")
    inputs = {ext: sha(scan.with_suffix(ext)) for ext in (".geom", ".f32")}
    pinned = json.loads((REPORT / "native-ab.json").read_text())["cases"]["full-corrected"]
    require(
        inputs[".geom"] == pinned["geom_sha256"]
        and inputs[".f32"] == pinned["projections_sha256"],
        "unexpected dataset",
    )
    reports.mkdir(parents=True, exist_ok=True)
    n = 512
    fov = 40
    batch = 16
    budget = 128
    scan_bytes = scan.with_suffix(".f32").stat().st_size
    evidence = dict(
        dataset="Alexander Meaney / University of Helsinki walnut, 720 views, 8x8 binning; CC BY 4.0; see NOTICE.md",
        inputs_sha256=inputs,
        list_sha256=sha(LIST),
        binaries_sha256={k: sha(BUILD / k) for k in ("cone-fdk.exe", "kernels.ptx")},
        n=n,
        fov_mm=fov,
        batch=batch,
        budget_mib=budget,
        protocol="One warmup each; six AB/BA pairs, same build, one GPU, no clock lock or cache eviction. Serial time sums six native subprocesses; packed time is one native process, including validated input, all payloads and durable publication. Python hashes, previews and extraction excluded. Read counts are application reads, not physical disk I/O.",
    )
    with TemporaryDirectory(dir=scratch) as folder:
        tmp = Path(folder)
        full = tmp / "full.f32"
        pack = tmp / "all.croi"
        full_meta = reconstruct(scan, full, n, fov, 768, batch)
        full_sha = sha(full)
        require(
            full_sha
            == json.loads((REPORT / "run-repeat.json").read_text())["baseline_outputs"][
                "walnut512"
            ],
            "full output changed from released reference",
        )
        evidence.update(
            parent_sha256=full_sha,
            gpu=full_meta["gpu"],
            driver_api=full_meta["driver_api"],
            cufft_version=full_meta["cufft_version"],
        )
        parent = np.memmap(full, "<f4", "r", shape=(n,) * 3)
        expected = [hashlib.sha256(parent[region(b)].tobytes()).hexdigest() for b in boxes]
        evidence["regions"] = [dict(box=b, parent_crop_sha256=h) for b, h in zip(boxes, expected)]

        def check(m, out, cap):
            require(
                m["views"] == 720
                and m["batch"] == batch
                and m["redzones_verified"]
                and m["peak_owned_bytes"] <= cap * 2**20
                and m["output_bytes"] == out.stat().st_size,
                "native batch, memory, length or guard check failed",
            )
            require(
                m["projection_bytes_read"] == m["projection_passes"] * scan_bytes
                and m["filter_batches"] == m["projection_passes"] * 45,
                "input pass counts disagree",
            )

        def serial():
            runs = []
            for i, s in enumerate(specs):
                out = tmp / f"r{i}.f32"
                m = reconstruct(scan, out, n, fov, budget, batch, roi=s)
                check(m, out, budget)
                require(
                    m["projection_passes"] == 1 and sha(out) == expected[i],
                    f"single ROI {i} differs from parent or pass count",
                )
                runs.append({k: m[k] for k in KEYS})
            return dict(
                individual=runs,
                **{
                    k: (max if k == "peak_owned_bytes" else sum)(m[k] for m in runs) for k in KEYS
                },
            )

        def packed(cap):
            m = reconstruct(scan, pack, n, fov, cap, batch, roi_list=LIST)
            check(m, pack, cap)
            with pack.open("rb") as f:
                idx = read_index(f)
                require(
                    idx["n"] == n
                    and idx["voxel_spacing_mm"] == fov / n
                    and idx["regions"] == m["regions"]
                    and len(idx["regions"]) == len(boxes),
                    "pack index mismatch",
                )
                for i, e in enumerate(idx["regions"]):
                    f.seek(e["offset_bytes"])
                    require(
                        e["origin_xyz"] + e["shape_xyz"] == boxes[i]
                        and hashlib.sha256(f.read(e["size_bytes"])).hexdigest() == expected[i],
                        f"packed ROI {i} differs from parent",
                    )
            return dict(
                **{k: m[k] for k in KEYS}, output_bytes=m["output_bytes"], sha256=sha(pack)
            )

        serial()
        warm = packed(budget)
        pairs = []
        for i in range(6):
            order = ("serial", "packed") if i % 2 == 0 else ("packed", "serial")
            pair = {k: (serial() if k == "serial" else packed(budget)) for k in order}
            require(
                pair["packed"]["projection_passes"] == 1
                and pair["packed"]["sha256"] == warm["sha256"],
                "repeatability or single-pass gate failed",
            )
            pairs.append(pair)
        a, b = (median(p[k]["process_seconds"] for p in pairs) for k in ("serial", "packed"))
        wins = sum(p["serial"]["process_seconds"] > p["packed"]["process_seconds"] for p in pairs)
        require(
            wins >= 5 and b < a * 0.9,
            "throughput gate failed (<10% reduction or fewer than five wins)",
        )
        pressure = packed(48)
        require(
            pressure["projection_passes"] > 1 and pressure["sha256"] == warm["sha256"],
            "multi-pass recovery under a 48 MiB cap failed",
        )
        preview(parent, pack, read_regions(pack), reports / "multi-roi-walnut.png")
        del parent
        evidence.update(
            pairs=pairs,
            serial_median_seconds=a,
            packed_median_seconds=b,
            speed_ratio=a / b,
            paired_wins=wins,
            pressure_48_mib=pressure,
        )
    (reports / "multi-roi-walnut.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf8"
    )
    print(
        f"6 boxes, 6 exact pairs: {a:.4f} -> {b:.4f} s; {a / b:.2f}x. 48 MiB: {pressure['projection_passes']} passes, identical pack."
    )


def read_regions(path):
    with path.open("rb") as f:
        return read_index(f)["regions"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=DATA / "full-corrected")
    p.add_argument("--reports", type=Path, default=REPORT)
    p.add_argument("--scratch", type=Path, default=None)
    a = p.parse_args()
    run(a.scan, a.reports, a.scratch)
