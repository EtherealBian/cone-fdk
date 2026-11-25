"""Qualify direct CUDA cross-sections on the pinned, center-corrected 720-view scan."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, DATA, REPORT, ROOT, Geometry, reconstruct
from compare_native import sha
from PIL import Image, ImageDraw
from spatial_oracle import backproject_points
from validate_roi import require

OBLIQUE = ROOT / "fixtures/walnut-oblique.txt"
REL_LIMIT, PEAK_LIMIT = 1e-5, 5e-5


def montage(images, labels, path):
    # One window for all four direct reconstructions; preserve pixel aspect (all pitches square here).
    low, high = np.percentile(images[0], [2, 99])
    canvas = Image.new("RGB", (1000, 312), "#152027")
    ink = ImageDraw.Draw(canvas)
    ink.text((14, 9), "HELSINKI WALNUT   /   FOUR DIRECT CUDA CROSS-SECTIONS", fill="#faf6ed")
    for i, (frame, label) in enumerate(zip(images, labels, strict=True)):
        image = Image.fromarray((255 * np.clip((frame - low) / (high - low), 0, 1)).astype("u1"))
        image.thumbnail((234, 234), Image.Resampling.LANCZOS)
        x = 14 + 246 * i
        canvas.paste(image, (x, 34 + (234 - image.height) // 2))
        ink.text((x, 273), label, fill="#f5d8a0")
    ink.text(
        (14, 293),
        "Shared intensity window. Oblique field: 32 x 28.8 mm; standard fields: 40 x 40 mm.   Data: A. Meaney / CC BY 4.0.",
        fill="#bfcad0",
    )
    canvas.save(path)


def run(scan, full, reports, scratch):
    hashes = {e: sha(scan.with_suffix(e)) for e in (".geom", ".f32")}
    pinned = json.loads((REPORT / "native-ab.json").read_text())["cases"]["full-corrected"]
    require(
        hashes == {".geom": pinned["geom_sha256"], ".f32": pinned["projections_sha256"]},
        "unexpected input scan",
    )
    reference = sha(full)
    require(
        reference
        == json.loads((REPORT / "run-repeat.json").read_text())["baseline_outputs"]["walnut512"],
        "full reference hash mismatch",
    )
    volume = np.memmap(full, "<f4", "r", shape=(512, 512, 512))
    g = Geometry.read(scan.with_suffix(".geom"))
    projections = np.memmap(scan.with_suffix(".f32"), "<f4", "r", shape=(g.a, g.v, g.u))
    results = dict(
        dataset="Helsinki walnut; 720 measured views, 8x8 binning, center corrected. See NOTICE.md.",
        input_sha256=hashes,
        parent_sha256=reference,
        parent_bytes=full.stat().st_size,
        binaries_sha256={n: sha(BUILD / n) for n in ("cone-fdk.exe", "kernels.ptx")},
        n=512,
        fov_mm=40,
        budget_mib=128,
        requested_batch=16,
        planes={},
    )
    pitch = 40 / 512
    images = []
    with TemporaryDirectory(dir=scratch) as tmp:
        temp = Path(tmp)
        dest = temp / "direct.f32"

        def direct(name, manifest):
            m = reconstruct(scan, dest, 512, 40, 128, 16, plane=manifest)
            shape = tuple(m["shape_yx"])
            require(
                m["format"] == "raw-f32-le-plane"
                and m["redzones_verified"]
                and m["peak_owned_bytes"] <= m["effective_budget_bytes"] == 128 << 20
                and m["output_bytes"] == dest.stat().st_size == 4 * np.prod(shape)
                and m["projection_passes"] == 1
                and m["projection_bytes_read"] == scan.with_suffix(".f32").stat().st_size,
                "wrong output layout, memory, read count or red zones",
            )
            image = np.fromfile(dest, "<f4").reshape(shape)
            require(np.isfinite(image).all(), "non-finite output")
            results["planes"][name] = {
                "manifest": manifest.read_text().splitlines(),
                "sha256": sha(dest),
                "native": m,
            }
            images.append(image)
            return image

        # Independent parent array selections; neither CUDA's affine nor a volume resampler creates the reference.
        for name, axis, index, col, row, ref in [
            ("axial", 2, 271, (pitch, 0, 0), (0, pitch, 0), volume[271, :, :]),
            ("coronal", 1, 231, (pitch, 0, 0), (0, 0, pitch), volume[:, 231, :]),
            ("sagittal", 0, 301, (0, pitch, 0), (0, 0, pitch), volume[:, :, 301]),
        ]:
            spec = temp / (name + ".txt")
            center = [0.0] * 3
            center[axis] = (index - 255.5) * pitch
            spec.write_text(
                "\n".join(
                    ["CFDKPLANE1 512 512", *(" ".join(map(str, x)) for x in [center, col, row])]
                )
                + "\n",
                encoding="ascii",
            )
            frame = direct(name, spec)
            require(
                np.array_equal(frame.view("<u4"), ref.view("<u4")),
                name + " differs from the published volume",
            )
            results["planes"][name].update(
                parent_index=index, bitwise_equal_to_parent=True, compared_pixels=frame.size
            )
        frame = direct("oblique", OBLIQUE)
        # Re-evaluate the specification independently, not from the GPU's reported transform.
        tag, w, h, *coords = OBLIQUE.read_text().split()
        require(tag == "CFDKPLANE1" and frame.shape == (int(h), int(w)), "oblique dimensions")
        center, col, row = np.array(coords, dtype="f8").reshape(3, 3)
        h, w = frame.shape
        rng = np.random.default_rng(7091)
        ij = [
            (y + int(rng.integers(min(16, h - y))), x + int(rng.integers(min(16, w - x))))
            for y in range(0, h, 16)
            for x in range(0, w, 16)
        ]
        jj, ii = np.unique(
            ij + [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (h // 2, w // 2)], axis=0
        ).T
        points = center + (ii[:, None] - (w - 1) / 2) * col + (jj[:, None] - (h - 1) / 2) * row
        ref, _ = backproject_points(projections, g, points)
        delta = frame[jj, ii].astype("f8") - ref
        rel = float(np.linalg.norm(delta) / np.linalg.norm(ref))
        peak = float(np.max(np.abs(delta)) / np.max(np.abs(ref)))
        require(
            rel < REL_LIMIT and peak < PEAK_LIMIT, f"oblique FP64 gate failed: {rel=}, {peak=}"
        )
        results["planes"]["oblique"].update(
            reference="FP64 spatial convolution, explicit ray/plane intersection and zero-extended bilinear interpolation, all 720 views; no parent-volume interpolation",
            sampling="One seeded pixel per 16x16 tile (NumPy PCG64 seed 7091), plus corners and center, deduplicated",
            compared_pixels=len(ii),
            relative_l2=rel,
            relative_l2_limit=REL_LIMIT,
            max_abs=float(np.max(np.abs(delta))),
            ref_peak=float(np.max(np.abs(ref))),
            max_abs_over_ref_peak=peak,
            max_abs_over_ref_peak_limit=PEAK_LIMIT,
        )
    reports.mkdir(parents=True, exist_ok=True)
    montage(
        images,
        [
            "AXIAL   /   Z = 271",
            "CORONAL   /   Y = 231",
            "SAGITTAL   /   X = 301",
            "OBLIQUE   /   0.10 mm per pixel",
        ],
        reports / "direct-planes.png",
    )
    (reports / "direct-planes.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8"
    )
    print(
        f"Three 512-square planes bitwise equal; {len(ii)} oblique FP64 checks: rel L2 {rel:.8g}, max/peak {peak:.8g}.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=DATA / "full-corrected")
    p.add_argument("--full", type=Path, default=DATA / "full-corrected-512.f32")
    p.add_argument("--reports", type=Path, default=REPORT)
    p.add_argument("--scratch", type=Path)
    a = p.parse_args()
    run(a.scan, a.full, a.reports, a.scratch)
