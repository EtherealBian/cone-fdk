"""Qualify oriented stacks on measured projections and an analytic, known-radius sphere."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, DATA, REPORT, ROOT, Geometry, reconstruct
from compare_native import sha
from phantom import SPHERE
from PIL import Image, ImageDraw
from spatial_oracle import backproject_points
from validate_roi import require

FIXTURE = ROOT / "fixtures/walnut-stack.txt"
# Center slice published before the stack implementation (commit 6ab0d3b).
FROZEN_CENTER = "1fd4b537d8716c58b1428368cdb012492896628395ceb7c0b185d23c3f0ef38f"


def write_spec(path, dims, vectors):
    tag = "CFDKSTACK1" if len(dims) == 3 else "CFDKPLANE1"
    path.write_text(
        "\n".join([" ".join(map(str, (tag, *dims))), *(" ".join(map(str, v)) for v in vectors)])
        + "\n",
        encoding="ascii",
    )
    return path


def pinned_inputs(scan, key):
    expected = json.loads((REPORT / "native-ab.json").read_bytes())["cases"][key]
    inputs = {e: sha(scan.with_suffix(e)) for e in (".geom", ".f32")}
    require(
        inputs == {".geom": expected["geom_sha256"], ".f32": expected["projections_sha256"]},
        "unexpected input scan",
    )
    return inputs


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf8", newline="\n")


def require_accounting(m, scan, out):
    require(
        m["output_bytes"] == out.stat().st_size
        and m["peak_owned_bytes"] <= m["effective_budget_bytes"] <= m["budget_bytes"]
        and m["redzones_verified"]
        and m["projection_bytes_read"]
        == scan.with_suffix(".f32").stat().st_size * m["projection_passes"],
        "size, read count, device budget or red-zone failure",
    )


def preview(a, step, path):
    lo, hi = np.percentile(a, [2, 99])
    frames = []
    for k, layer in enumerate(a):
        image = Image.fromarray((255 * np.clip((layer - lo) / (hi - lo), 0, 1)).astype("u1"))
        canvas = Image.new("RGB", (344, 355), "#142027")
        canvas.paste(image, (12, 31))
        ink = ImageDraw.Draw(canvas)
        ink.text((12, 9), "WALNUT   /   OBLIQUE STACK", fill="#eee9dc")
        ink.text((244, 9), f"{(k - (len(a) - 1) / 2) * step:+.2f} mm", fill="#f5c681")
        ink.text((12, 327), "A. Meaney / CC BY 4.0   |   fixed intensity window", fill="#cad5db")
        frames.append(canvas)
    frames[0].save(
        path, save_all=True, append_images=frames[1:] + frames[-2:0:-1], duration=130, loop=0
    )


def check_reader(path, a, origin, steps):
    # Optional acceptance-only dependency; the native NRRD writer has none.
    from importlib.metadata import version

    import nrrd

    got, fields = nrrd.read(str(path), index_order="C")
    require(
        np.array_equal(got.view("<u4"), a.view("<u4"))
        and fields["space"] == "3D-right-handed"
        and fields["space units"] == ["mm"] * 3
        and list(fields["sizes"]) == list(reversed(a.shape)),
        "external NRRD payload or dimensions",
    )
    np.testing.assert_array_equal(fields["space directions"], steps)
    np.testing.assert_allclose(fields["space origin"], origin, rtol=0, atol=1e-13)
    return dict(
        reader="pynrrd",
        version=version("pynrrd"),
        payload_bitwise_equal=True,
        space_origin_mm=fields["space origin"].tolist(),
        directions_and_units_checked=True,
    )


def walnut(scan, full, reports, tmp, external):
    inputs = pinned_inputs(scan, "full-corrected")
    full_hash = sha(full)
    require(
        full_hash
        == json.loads((REPORT / "run-repeat.json").read_bytes())["baseline_outputs"]["walnut512"],
        "wrong parent volume",
    )
    g = Geometry.read(scan.with_suffix(".geom"))
    projections = np.memmap(scan.with_suffix(".f32"), "<f4", "r", shape=(g.a, g.v, g.u))
    tag, w, h, d, *v = FIXTURE.read_text().split()
    w, h, d = map(int, (w, h, d))
    center, col, row, step = np.array(v, dtype="f8").reshape(4, 3)
    require(tag == "CFDKSTACK1" and d % 2 == 1, "expected odd-depth fixture")
    out = tmp / "stack.f32"
    spec = tmp / "spec.txt"
    main = reconstruct(scan, out, 512, 40, 128, 16, stack=FIXTURE)
    require_accounting(main, scan, out)
    require(
        main["projection_passes"] == 1
        and main["shape_zyx"] == [d, h, w]
        and main["format"] == "raw-f32-le-stack",
        "unexpected stack shape or passes",
    )
    digest = sha(out)
    a = np.fromfile(out, "<f4").reshape(d, h, w)
    require(np.isfinite(a).all(), "nonfinite payload")
    center_hash = hashlib.sha256(a[d // 2]).hexdigest()
    require(center_hash == FROZEN_CENTER, "center slice changed")
    serial_bytes = 0
    single = tmp / "plane.f32"
    for k, layer in enumerate(a):
        at = center + (k - (d - 1) / 2) * step
        write_spec(spec, (w, h), (at, col, row))
        m = reconstruct(scan, single, 512, 40, 128, 16, plane=spec)
        require_accounting(m, scan, single)
        require(
            m["projection_passes"] == 1 and single.read_bytes() == layer.tobytes(),
            f"difference at layer {k}",
        )
        serial_bytes += m["projection_bytes_read"]
    policies = {}
    for policy in ("batch-first", "min-passes"):
        m = reconstruct(scan, out, 512, 40, 48, 16, stack=FIXTURE, planner=policy)
        require_accounting(m, scan, out)
        require(sha(out) == digest, "budget or policy changed payload")
        policies[policy] = m
    require(
        policies["batch-first"]["projection_passes"] > 1
        and policies["batch-first"]["rows_per_pass"] % h
        and policies["min-passes"]["projection_passes"] == 1,
        "expected interior seam and a single-pass alternative at 48 MiB",
    )
    nrrd_path = tmp / "tilted.nrrd"
    nrrd_metrics = reconstruct(
        scan, nrrd_path, 512, 40, 128, 16, stack=FIXTURE, stack_format="nrrd"
    )
    require_accounting(nrrd_metrics, scan, nrrd_path)
    header, payload = nrrd_path.read_bytes().split(b"\n\n", 1)
    require(
        len(header) + 2 == nrrd_metrics["data_offset_bytes"]
        and hashlib.sha256(payload).hexdigest() == digest,
        "NRRD header offset or payload mismatch",
    )
    nrrd_evidence = dict(
        header=header.decode("ascii").splitlines(),
        file_sha256=sha(nrrd_path),
        payload_sha256=digest,
        native=nrrd_metrics,
    )
    if external:
        nrrd_evidence["round_trip"] = check_reader(
            nrrd_path,
            a,
            center - ((w - 1) * col + (h - 1) * row + (d - 1) * step) / 2,
            [col, row, step],
        )
    # Every layer: 4 corners, a center and one seeded point per 64x64 tile.
    rng = np.random.default_rng(81573)
    coords = []
    for k in range(d):
        pairs = {(0, 0), (h - 1, 0), (0, w - 1), (h - 1, w - 1), (h // 2, w // 2)}
        for y in range(0, h, 64):
            for x in range(0, w, 64):
                pairs.add(
                    (int(rng.integers(y, min(h, y + 64))), int(rng.integers(x, min(w, x + 64))))
                )
        coords.extend((k, y, x) for y, x in sorted(pairs))
    kk, jj, ii = np.array(coords).T
    points = (
        center
        + (ii[:, None] - (w - 1) / 2) * col
        + (jj[:, None] - (h - 1) / 2) * row
        + (kk[:, None] - (d - 1) / 2) * step
    )
    ref, _ = backproject_points(projections, g, points)
    delta = a[kk, jj, ii].astype("f8") - ref
    rel = float(np.linalg.norm(delta) / np.linalg.norm(ref))
    peak = float(np.max(np.abs(delta)) / np.max(np.abs(ref)))
    require(rel < 1e-5 and peak < 5e-5, f"FP64 stack oracle: {rel=}, {peak=}")
    # Permute and reverse parent axes to make transposition or a wrong origin visible.
    pitch = 40 / 512
    dims = (61, 54, 9)
    first = np.array([222, 263, 231])
    steps = pitch * np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]])
    center2 = (first - 255.5) * pitch + ((np.array(dims) - 1) / 2) @ steps
    write_spec(spec, dims, [center2, *steps])
    crop = reconstruct(scan, out, 512, 40, 128, 16, stack=spec)
    require_accounting(crop, scan, out)
    volume = np.memmap(full, "<f4", "r", shape=(512, 512, 512))
    expected = volume[231:285, 203:264, 222:231].transpose(2, 0, 1)[:, :, ::-1]
    require(out.read_bytes() == expected.tobytes(), "stack differs from reindexed parent")
    preview(a, np.linalg.norm(step), reports / "oblique-stack.gif")
    save_json(
        reports / "oblique-stack.json",
        dict(
            dataset="Helsinki walnut; 720 measured views, 8x8 binning, center corrected. See NOTICE.md.",
            inputs_sha256=inputs,
            binaries_sha256={name: sha(BUILD / name) for name in ("cone-fdk.exe", "kernels.ptx")},
            manifest=FIXTURE.read_text().splitlines(),
            stack_sha256=digest,
            main=main,
            serial_planes=dict(
                count=d,
                compared_voxels=a.size,
                bitwise_equal=True,
                projection_bytes_read=serial_bytes,
                frozen_center_sha256=center_hash,
            ),
            low_budget_48_mib=policies,
            nrrd=nrrd_evidence,
            fp64=dict(
                sampling="Each layer: corners, center and one seeded point per 64x64 tile. Seed 81573, all views; independent spatial convolution and explicit ray-plane intersections.",
                compared_points=len(coords),
                relative_l2=rel,
                relative_l2_limit=1e-5,
                max_abs_over_ref_peak=peak,
                max_abs_over_ref_peak_limit=5e-5,
            ),
            permuted_parent_crop=dict(
                parent_sha256=full_hash,
                manifest=spec.read_text().splitlines(),
                compared_voxels=expected.size,
                bitwise_equal=True,
                sha256=sha(out),
            ),
        ),
    )
    print(
        f"Walnut: {d} serial planes matched; {len(coords)} FP64 samples, rel L2 {rel:.8g}, max/peak {peak:.8g}.",
        flush=True,
    )


def sphere(scan, reports, tmp):
    inputs = pinned_inputs(scan, "phantom")
    w, h, d = 91, 81, 67
    spacing = np.array([0.06, 0.07, 0.08])
    axes = np.array([[0.8, 0.6, 0], [-0.36, 0.48, 0.8], [-0.48, 0.64, -0.6]]) * spacing[:, None]
    true_center = np.array(SPHERE[:3])
    center = true_center + np.array([0.16, -0.21, 0.12])
    volume = 4 * np.pi * SPHERE[3] ** 3 / 3
    spec = write_spec(tmp / "sphere.txt", (w, h, d), (center, *axes))
    out = tmp / "sphere.f32"
    m = reconstruct(scan, out, 512, 32, 32, 16, stack=spec)
    require_accounting(m, scan, out)
    threshold = 0.04 + SPHERE[4] / 2
    k, j, i = np.nonzero(np.fromfile(out, "<f4").reshape(d, h, w) > threshold)
    require(
        len(i) > 0
        and all(min(idx) > 0 and max(idx) < n - 1 for idx, n in [(i, w), (j, h), (k, d)]),
        "sphere mask empty or clipped",
    )
    measured = len(i) * abs(np.linalg.det(axes))
    centroid = (
        center + (np.array([i.mean(), j.mean(), k.mean()]) - (np.array([w, h, d]) - 1) / 2) @ axes
    )
    err = abs(measured / volume - 1)
    distance = np.linalg.norm(centroid - true_center)
    dia = (6 * measured / np.pi) ** (1 / 3)
    dia_err = abs(dia - 2 * SPHERE[3])
    require(
        err < 0.03 and distance < 0.05 and dia_err < 0.05,
        f"analytic 3D measurement: {err=}, {distance=}, {dia_err=}",
    )
    save_json(
        reports / "oblique-stack-phantom.json",
        dict(
            dataset="Analytic 180-view phantom, known-radius inclusion; not a measured part",
            inputs_sha256=inputs,
            manifest=spec.read_text().splitlines(),
            stack_sha256=sha(out),
            native=m,
            method="Fixed half-contrast threshold, all voxels, no registration or fitted intensity. Volume = voxel count times abs(det(step matrix)); centroid of included voxel centers.",
            threshold_per_mm=threshold,
            mask_voxels=len(i),
            centroid_xyz_mm=centroid.tolist(),
            true_center_xyz_mm=true_center.tolist(),
            center_distance_mm=distance,
            center_limit_mm=0.05,
            expected_volume_mm3=volume,
            observed_volume_mm3=measured,
            volume_relative_error=err,
            volume_relative_error_limit=0.03,
            equivalent_diameter_mm=dia,
            true_diameter_mm=2 * SPHERE[3],
            diameter_abs_error_mm=dia_err,
            diameter_limit_mm=0.05,
        ),
    )
    print(
        f"Analytic sphere: volume error {err:.3%}, centroid error {distance:.5f} mm, equivalent diameter error {dia_err:.5f} mm.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", choices=["walnut", "phantom"])
    p.add_argument("--scan", type=Path)
    p.add_argument("--full", type=Path, default=DATA / "full-corrected-512.f32")
    p.add_argument("--reports", type=Path, default=REPORT)
    p.add_argument(
        "--check-pynrrd",
        action="store_true",
        help="also verify the walnut NRRD with an installed external reader",
    )
    a = p.parse_args()
    a.reports.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as scratch:
        if a.case == "walnut":
            walnut(
                a.scan or DATA / "full-corrected",
                a.full,
                a.reports,
                Path(scratch),
                a.check_pynrrd,
            )
        else:
            sphere(a.scan or DATA / "phantom", a.reports, Path(scratch))
