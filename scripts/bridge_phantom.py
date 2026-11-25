"""Generate the preregistered bridge experiment in detector space, never from scored wall masks."""

import argparse
from pathlib import Path

import numpy as np
from bridge_design import SPEC_SHA256, protocol
from common import Geometry
from compare_native import sha
from loss_phantom import MU, PHOTONS, counting_scan
from remount_loss import write_doc
from remount_phantom import BALLS, pose
from specimen_phantom import clip_z, interval
from witness_phantom import INSERTS


def chords(src, direction, spec, depth):
    """Material path in ray-parameter units; half-open adjacent slabs own a parallel ray once."""
    sx, sy, sz = src
    dx, dy, dz = direction
    tube = spec["tube_mm"]
    ro, ri = tube["outer_radius"], tube["inner_radius"]
    outer, bore = [interval(sx, sy, dx, dy, 0, 0, r) for r in (ro, ri)]
    z0, z1 = tube["z_min"], tube["z_max"]
    path = clip_z(outer, sz, dz, z0, z1) - clip_z(bore, sz, dz, z0, z1)
    if depth is not None:
        phi = np.deg2rad(spec["flat_normal_deg"])
        nx, ny = np.cos(phi), np.sin(phi)
        slope = nx * dx + ny * dy
        parallel = np.abs(slope) <= 1e-12
        breaks = spec["axial_breaks_mm"]
        for lo, hi, d in zip(
            breaks[:-1],
            breaks[1:],
            (spec["shoulder_depth_mm"], depth, spec["shoulder_depth_mm"]),
            strict=True,
        ):
            if not 0 <= d < ro - ri:
                raise ValueError("cut must not intersect bore")
            offset = ro - d - nx * sx - ny * sy
            cut = offset / np.where(parallel, 1, slope)
            piece = (
                np.where(slope > 1e-12, np.maximum(outer[0], cut), outer[0]),
                np.where(slope < -1e-12, np.minimum(outer[1], cut), outer[1]),
            )
            # clip_z includes both ends for parallel rays. Our touching slabs are [lo, hi).
            removed = clip_z(piece, sz, dz, lo, hi)
            path -= np.where(
                (~parallel | (offset <= 0)) & ~((np.abs(dz) < 1e-12) & (sz == hi)), removed, 0
            )
    if not np.isfinite(path).all() or np.min(path) < -1e-8:
        raise ValueError("invalid material chord")
    return np.maximum(path, 0)


def project(prefix, spec, mount, depth):
    a = spec["acquisition"]
    g = Geometry(
        *(
            a[k]
            for k in (
                "angles",
                "detector_rows",
                "detector_cols",
                "du_mm",
                "dv_mm",
                "source_axis_mm",
                "source_detector_mm",
            )
        )
    )
    g.save(prefix.with_suffix(".geom"))
    rot, t = pose(mount)
    u = (np.arange(g.u) - (g.u - 1) / 2) * g.du / g.D
    v = ((np.arange(g.v) - (g.v - 1) / 2) * g.dv / g.D)[:, None]
    offsets = a["subray_offsets_bins"]
    with prefix.with_suffix(".f32").open("xb") as dst:
        for theta in g.angles():
            sn, cs = np.sin(theta), np.cos(theta)
            src = (np.array([g.R * sn, -g.R * cs, 0]) - t) @ rot
            view = np.zeros((g.v, g.u))
            for off in offsets:
                uu = u + off * g.du / g.D
                direction = [
                    (-sn + cs * uu) * x + (cs + sn * uu) * y + v * z for x, y, z in rot.T
                ]
                dx, dy, dz = direction
                n2 = dx * dx + dy * dy + dz * dz
                signal = MU * chords(src, direction, spec, depth) * np.sqrt(n2)
                for centers, r, mu in ((BALLS, 0.65, 0.24), (INSERTS, 0.50, 0.06)):
                    for x, y, z in src - centers:
                        cross = (
                            (y * dz - z * dy) ** 2
                            + (z * dx - x * dz) ** 2
                            + (x * dy - y * dx) ** 2
                        )
                        signal += 2 * mu * np.sqrt(np.maximum(0, r * r - cross / n2))
                view += signal / len(offsets)
            view.astype("<f4").tofile(dst)


def generate(out):
    spec = protocol()
    if spec["acquisition"]["photons_per_air_bin"] != PHOTONS:
        raise ValueError("counting model differs from protocol")
    out.mkdir(exist_ok=False, parents=True)
    raw = {}
    records = {}
    for scan in spec["scans"]:
        key = (scan["mount"], scan["neck_depth_mm"])
        if key not in raw:
            clean = out / ("clean-" + scan["id"])
            project(clean, spec, *key)
            raw[key] = clean
        noisy = out / scan["id"]
        counting_scan(raw[key], noisy, scan["seed"])
        records[scan["id"]] = {ext: sha(noisy.with_suffix("." + ext)) for ext in ("geom", "f32")}
        print("projected", scan["id"], flush=True)
    write_doc(out / "sources.json", dict(study_sha256=SPEC_SHA256, scans=records))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    generate(p.parse_args().output)
