"""Finite, remounted tubes of different wall sizes. Truth is for simulation/scoring only."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from common import Geometry
from loss_phantom import MU, counting_scan
from remount_loss import write_doc
from remount_phantom import BALLS, pose
from witness_phantom import INSERTS


@dataclass(frozen=True)
class Tube:
    outer: float
    inner: float
    # Outside plane cut: azimuth degrees, depth below OD, lower/upper axial extent.
    flat: tuple
    # Bore-connected cylindrical pocket: azimuth, center radius, tool radius, Z bounds.
    pocket: tuple | None
    seeds: tuple


TUBES = {
    "seam-flat": Tube(9.05, 6.30, (358.0, 0.78, -2.7, 0.7), None, (7413, 8634, 9841)),
    "dual-band": Tube(
        9.68, 5.80, (63.0, 0.66, -3.3, -1.1), (218.0, 4.83, 1.70, 0.5, 2.9), (12781, 13567, 14551)
    ),
}


def clip_z(interval, src_z, dz, lo, hi):
    """Length in ray-parameter units including axial-parallel rays and empty intersections."""
    parallel = np.abs(dz) < 1e-12
    d = np.where(parallel, 1.0, dz)
    a, b = (lo - src_z) / d, (hi - src_z) / d
    left, right = interval
    cut = np.maximum(0, np.minimum(right, np.maximum(a, b)) - np.maximum(left, np.minimum(a, b)))
    return np.where(parallel, np.maximum(0, right - left) if lo <= src_z <= hi else 0.0, cut)


def interval(sx, sy, dx, dy, cx, cy, radius):
    x, y = sx - cx, sy - cy
    a = dx * dx + dy * dy
    mid = -(x * dx + y * dy) / a
    half = np.sqrt(np.maximum(0.0, radius**2 * a - (x * dy - y * dx) ** 2)) / a
    return mid - half, mid + half


def removal(tube, src, direction, outer, bore):
    sx, sy, sz = src
    dx, dy, dz = direction
    phi, depth, z0, z1 = tube.flat
    nx, ny = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
    m = nx * dx + ny * dy
    b = tube.outer - depth - nx * sx - ny * sy
    cut = b / np.where(np.abs(m) > 1e-12, m, 1)
    piece = (
        np.where(m > 1e-12, np.maximum(outer[0], cut), outer[0]),
        np.where(m < -1e-12, np.minimum(outer[1], cut), outer[1]),
    )
    removed = np.where((np.abs(m) > 1e-12) | (b <= 0), clip_z(piece, sz, dz, z0, z1), 0.0)
    if tube.pocket:
        # Pockets are wholly inside the OD, intersect the bore and are disjoint in Z from the flat.
        phi, e, r, z0, z1 = tube.pocket
        assert abs(e - r) < tube.inner < e + r < tube.outer and z0 > tube.flat[3]
        pocket = interval(
            sx, sy, dx, dy, e * np.cos(np.deg2rad(phi)), e * np.sin(np.deg2rad(phi)), r
        )
        common = np.maximum(pocket[0], bore[0]), np.minimum(pocket[1], bore[1])
        removed += clip_z(pocket, sz, dz, z0, z1) - clip_z(common, sz, dz, z0, z1)
    return removed


def project(path, tube, mount, changed):
    g = Geometry(360, 144, 480, 0.13, 0.25, 120.0, 205.0)
    g.save(path.with_suffix(".geom"))
    rot, t = pose(mount)
    u = (np.arange(g.u) - (g.u - 1) / 2) * g.du / g.D
    v = ((np.arange(g.v) - (g.v - 1) / 2) * g.dv / g.D)[:, None]
    with path.with_suffix(".f32").open("xb") as dst:
        for theta in g.angles():
            sn, cs = np.sin(theta), np.cos(theta)
            src = (np.array([g.R * sn, -g.R * cs, 0]) - t) @ rot
            sx, sy, sz = src
            accum = np.zeros((g.v, g.u))
            for off in (-1 / 3, 0, 1 / 3):
                uu = u + off * g.du / g.D
                d = [(-sn + cs * uu) * x + (cs + sn * uu) * y + v * z for x, y, z in rot.T]
                dx, dy, dz = d
                outer = interval(sx, sy, dx, dy, 0.0, 0.0, tube.outer)
                bore = interval(sx, sy, dx, dy, 0.0, 0.0, tube.inner)
                material = clip_z(outer, sz, dz, -7, 7) - clip_z(bore, sz, dz, -7, 7)
                if changed:
                    material -= removal(tube, src, d, outer, bore)
                if material.min() < -1e-8:
                    raise ValueError("invalid analytic material chord")
                norm2 = dx * dx + dy * dy + dz * dz
                signal = MU * np.maximum(0.0, material) * np.sqrt(norm2)
                for group, r, mu in ((BALLS, 0.65, 0.24), (INSERTS, 0.50, 0.06)):
                    for x, y, z in src - group:
                        cross = (
                            (y * dz - z * dy) ** 2
                            + (z * dx - x * dz) ** 2
                            + (x * dy - y * dx) ** 2
                        )
                        signal += 2 * mu * np.sqrt(np.maximum(0.0, r * r - cross / norm2))
                accum += signal / 3
            accum.astype("<f4").tofile(dst)


def generate(out):
    out.mkdir(parents=True, exist_ok=False)
    for name, tube in TUBES.items():
        job = out / name
        job.mkdir()
        for k, (role, mount, changed) in enumerate(
            [
                ("reference", "first", False),
                ("change", "second", True),
                ("repeat", "second", False),
            ]
        ):
            if role == "repeat" and name != "seam-flat":
                continue
            project(job / role, tube, mount, changed)
            counting_scan(job / role, job / ("noisy-" + role), tube.seeds[k])
            print(name, role, flush=True)
    write_doc(out / "design.json", {k: asdict(v) for k, v in TUBES.items()})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    generate(cli.parse_args().output)
