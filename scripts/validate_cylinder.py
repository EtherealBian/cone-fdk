"""Qualify cylindrical FDK against an analytic seam-crossing target and measured projections."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from common import BUILD, DATA, REPORT, ROOT, Geometry, reconstruct
from compare_native import sha
from phantom import SPHERE
from spatial_oracle import backproject_points
from validate_roi import require
from validate_stack import pinned_inputs, require_accounting, save_json

FIXTURE = ROOT / "fixtures/walnut-cylinder.txt"


def spec(path, dims, center, axis, seam, steps):
    path.write_text(
        "CFDKCYL1 "
        + "\n".join(" ".join(map(str, v)) for v in (dims, center, axis, seam, steps))
        + "\n",
        encoding="ascii",
    )
    return path


def sphere(scan, reports, tmp):
    inputs = pinned_inputs(scan, "phantom")
    cx, cy, cz, ball, contrast = SPHERE
    r0 = np.hypot(cx, cy)
    a0 = np.arctan2(cy, cx)
    seam = a0 - np.deg2rad(2)
    pitch = 0.05
    w, h, d = 720, 101, 3
    center_z = cz + 0.2
    first = r0 - 0.8
    dr = 0.8
    path = spec(
        tmp / "spec",
        (w, h, d),
        (0, 0, center_z),
        (0, 0, 1),
        (np.cos(seam), np.sin(seam), 0),
        (first, dr, pitch),
    )
    out = tmp / "surface.f32"
    m = reconstruct(scan, out, 512, 32, 32, 16, cylinder=path)
    require_accounting(m, scan, out)
    maps = np.fromfile(out, "<f4").reshape(d, h, w)
    threshold = 0.04 + contrast / 2
    angle = 2 * np.pi * np.arange(w) / w
    z = center_z + (np.arange(h) - (h - 1) / 2) * pitch
    nearest = int(np.argmin(abs(z - cz)))
    require(abs(z[nearest] - cz) < 1e-12, "center row alignment")
    rows = []
    for k, layer in enumerate(maps):
        r = first + k * dr
        mask = layer > threshold
        j, i = np.nonzero(mask)
        require(
            len(i) > 0
            and mask[:, 0].any()
            and mask[:, -1].any()
            and not (mask[0].any() or mask[-1].any()),
            "empty, clipped or non-straddling sphere",
        )
        phase = np.angle(np.mean(np.exp(1j * angle[i])))
        da = np.angle(np.exp(1j * (phase + seam - a0)))
        dz = float(z[j].mean() - cz)
        surface_err = np.hypot(r * da, dz)
        exact = 2 * r * np.arccos((r * r + r0 * r0 - ball * ball) / (2 * r * r0))
        measured = mask[nearest].sum() * 2 * np.pi * r / w
        transitions = np.count_nonzero(mask[nearest] != np.roll(mask[nearest], 1))
        require(
            transitions == 2 and surface_err < 0.06 and abs(measured - exact) < 0.15,
            f"sphere ruler: {transitions=}, {surface_err=}, {measured=}, {exact=}",
        )
        rows.append(
            dict(
                radius_mm=r,
                mask_samples=len(i),
                crosses_seam=True,
                angular_error_deg=np.rad2deg(da),
                axial_error_mm=dz,
                surface_center_error_mm=surface_err,
                true_arc_mm=exact,
                measured_arc_mm=measured,
                arc_abs_error_mm=abs(measured - exact),
            )
        )
    save_json(
        reports / "cylinder-phantom.json",
        dict(
            dataset="Analytic sphere, not a measured part",
            inputs_sha256=inputs,
            manifest=path.read_text().splitlines(),
            native=m,
            sha256=sha(out),
            threshold_per_mm=threshold,
            target_azimuth_deg=2,
            center_limit_mm=0.06,
            arc_limit_mm=0.15,
            per_shell=rows,
        ),
    )
    print(
        "Seam-crossing sphere:",
        [(round(q["surface_center_error_mm"], 5), round(q["arc_abs_error_mm"], 5)) for q in rows],
        flush=True,
    )


def walnut(scan, reports, tmp):
    from cylinder_atlas import write_atlas

    inputs = pinned_inputs(scan, "full-corrected")
    g = Geometry.read(scan.with_suffix(".geom"))
    p = np.memmap(scan.with_suffix(".f32"), "<f4", "r", shape=(g.a, g.v, g.u))
    tag, w, h, d, *v = FIXTURE.read_text().split()
    w, h, d = map(int, (w, h, d))
    center, axis, seam, steps = np.array(v, dtype="f8").reshape(4, 3)
    first, dr, pitch = steps
    require(tag == "CFDKCYL1" and w % 4 == 0, "fixture must support quarter turn")
    out = tmp / "sheet.f32"
    m = reconstruct(scan, out, 512, 40, 128, 16, cylinder=FIXTURE)
    require_accounting(m, scan, out)
    require(
        m["projection_passes"] == 1
        and m["shape_rha"] == [d, h, w]
        and m["format"] == "raw-f32-le-cylinder",
        "native atlas layout",
    )
    digest = sha(out)
    a = np.fromfile(out, "<f4").reshape(d, h, w)
    tangent = np.cross(axis, seam)
    path = spec(tmp / "rotated", (w, h, d), center, axis, tangent, steps)
    rot = reconstruct(scan, out, 512, 40, 128, 16, cylinder=path)
    require_accounting(rot, scan, out)
    require(
        np.roll(a, -w // 4, axis=2).tobytes() == out.read_bytes(), "quarter-turn seam mismatch"
    )
    low = {}
    for policy in ("batch-first", "min-passes"):
        lm = reconstruct(scan, out, 512, 40, 46, 16, cylinder=FIXTURE, planner=policy)
        require_accounting(lm, scan, out)
        require(sha(out) == digest, "streamed payload changed")
        low[policy] = lm
    require(
        low["batch-first"]["projection_passes"] > 1
        and low["batch-first"]["rows_per_pass"] % h != 0
        and low["min-passes"]["projection_passes"] == 1,
        "expected split inside a shell and a one-pass alternative",
    )
    # Fix the sample design before reconstruction; include both edges of every sheet.
    rng = np.random.default_rng(37091)
    coords = []
    for k in range(d):
        pairs = {
            (j, i) for j in (0, h // 2, h - 1) for i in (0, 1, w // 4, w // 2, 3 * w // 4, w - 1)
        }
        for y in range(0, h, 32):
            for x in range(0, w, 64):
                pairs.add(
                    (int(rng.integers(y, min(y + 32, h))), int(rng.integers(x, min(x + 64, w))))
                )
        coords.extend((k, j, i) for j, i in sorted(pairs))
    kk, jj, ii = np.array(coords).T
    theta = (ii * 2 * np.pi / w)[:, None]
    r = (first + dr * kk)[:, None]
    pts = (
        center
        + ((jj - (h - 1) / 2) * pitch)[:, None] * axis
        + r * (np.cos(theta) * seam + np.sin(theta) * tangent)
    )
    ref, _ = backproject_points(p, g, pts)
    dif = a[kk, jj, ii] - ref
    rel = np.linalg.norm(dif) / np.linalg.norm(ref)
    peak = abs(dif).max() / abs(ref).max()
    require(rel < 1e-5 and peak < 5e-5, f"cylinder oracle: {rel=}, {peak=}")
    preview = write_atlas(a, m, reports)
    save_json(
        reports / "cylinder-walnut.json",
        dict(
            dataset="Helsinki walnut, corrected, 720 measured views. See NOTICE.md.",
            inputs_sha256=inputs,
            binaries_sha256={k: sha(BUILD / k) for k in ("cone-fdk.exe", "kernels.ptx")},
            manifest=FIXTURE.read_text().splitlines(),
            native=m,
            payload_sha256=digest,
            quarter_turn=dict(bitwise_equal=True, samples=a.size, shift_columns=w // 4),
            low_budget_46_mib=low,
            fp64=dict(
                samples=len(coords),
                relative_l2=rel,
                relative_l2_limit=1e-5,
                max_abs_over_ref_peak=peak,
                max_abs_over_ref_peak_limit=5e-5,
                seed=37091,
            ),
            preview=preview,
        ),
    )
    print(
        f"Walnut: {a.size} samples identical after seam rotation; {len(coords)} FP64 points, relative L2 {rel:.8g}.",
        flush=True,
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("case", choices=["phantom", "walnut"])
    cli.add_argument("--scan", type=Path)
    cli.add_argument("--reports", type=Path, default=REPORT)
    a = cli.parse_args()
    a.reports.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as scratch:
        if a.case == "phantom":
            sphere(a.scan or DATA / "phantom", a.reports, Path(scratch))
        else:
            walnut(a.scan or DATA / "full-corrected", a.reports, Path(scratch))
