"""Analytic radiographs of a marked finite tube; never pass generating poses to the locator."""

import argparse
from pathlib import Path

import numpy as np
from common import Geometry
from loss_phantom import FLAT, MU, RELIEF, RI, RO, counting_scan
from scipy.spatial.transform import Rotation

BALLS = np.array([[-11, -6, -3], [11, -4, 2], [-4, 11, 3], [7, 9, -3]], dtype=float)
POSES = {"first": ((-7, 3, -4), (-0.35, 0.25, -0.6)), "second": ((61, -9, 7), (0.75, -0.55, 1.1))}


def pose(key):
    angles, t = POSES[key]
    return Rotation.from_euler("zyx", angles, degrees=True).as_matrix(), np.asarray(t)


DEEP_FLAT = (0.0, 1.40, FLAT[2], FLAT[3])


def project(scan, mount, *, relief, missing=False, deep=False):
    rot, t = pose(mount)
    g = Geometry(360, 144, 480, 0.13, 0.25, 120.0, 205.0)
    g.save(scan.with_suffix(".geom"))
    u = (np.arange(g.u) - (g.u - 1) / 2) * g.du / g.D
    v = (np.arange(g.v)[:, None] - (g.v - 1) / 2) * g.dv / g.D
    balls = BALLS[:-1] if missing else BALLS
    with scan.with_suffix(".f32").open("wb") as file:
        for theta in g.angles():
            s, c = np.sin(theta), np.cos(theta)
            src = (np.array([g.R * s, -g.R * c, 0]) - t) @ rot
            view = np.zeros((g.v, g.u))
            for offset in (-1 / 3, 0, 1 / 3):
                uu = u + offset * g.du / g.D
                sx, sy, sz = src
                d = [(-s + c * uu) * x + (c + s * uu) * y + v * z for x, y, z in rot.T]
                dx, dy, dz = d
                q = dx**2 + dy**2
                norm2 = q + dz**2

                def radial(x, y, r, dx=dx, dy=dy, q=q):
                    mid = -(x * dx + y * dy) / q
                    half = np.sqrt(np.maximum(0, q * r * r - (x * dy - y * dx) ** 2)) / q
                    return mid - half, mid + half

                def slab(interval, lo, hi, dz=dz, sz=sz):
                    parallel = abs(dz) < 1e-12
                    denom = np.where(parallel, 1, dz)
                    a, b = (lo - sz) / denom, (hi - sz) / denom
                    left, right = interval
                    out = np.maximum(
                        0,
                        np.minimum(right, np.maximum(a, b)) - np.maximum(left, np.minimum(a, b)),
                    )
                    return np.where(
                        parallel, np.maximum(0, right - left) if lo <= sz <= hi else 0, out
                    )

                outer, bore = radial(sx, sy, RO), radial(sx, sy, RI)
                length = slab(outer, -7, 7) - slab(bore, -7, 7)
                # Existing flat, normal +material X, separate from the change of interest.
                _, depth, lo, hi = DEEP_FLAT if deep else FLAT
                edge = RO - depth - sx
                cut = edge / np.where(abs(dx) > 1e-12, dx, 1)
                trimmed = (
                    np.where(dx > 1e-12, np.maximum(outer[0], cut), outer[0]),
                    np.where(dx < -1e-12, np.minimum(outer[1], cut), outer[1]),
                )
                length -= np.where((abs(dx) > 1e-12) | (edge <= 0), slab(trimmed, lo, hi), 0)
                if relief:
                    phi, e, r, lo, hi = RELIEF
                    phi = np.deg2rad(phi)
                    pocket = radial(sx - e * np.cos(phi), sy - e * np.sin(phi), r)
                    overlap = (np.maximum(bore[0], pocket[0]), np.minimum(bore[1], pocket[1]))
                    length -= slab(pocket, lo, hi) - slab(overlap, lo, hi)
                signal = MU * length
                for x, y, z in src - balls:
                    # Squared ray/center distance via cross product, not voxelization.
                    cross2 = (
                        (y * dz - z * dy) ** 2 + (z * dx - x * dz) ** 2 + (x * dy - y * dx) ** 2
                    )
                    signal += 0.24 * 2 * np.sqrt(np.maximum(0, 0.65**2 * norm2 - cross2)) / norm2
                view += signal * np.sqrt(norm2) / 3
            view.astype("<f4").tofile(file)


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, mount, pocket, missing in [
        ("reference", "first", False, False),
        ("repeat", "second", False, False),
        ("changed", "second", True, False),
        ("missing", "second", True, True),
        ("deepened", "second", False, False),
    ]:
        project(root / name, mount, relief=pocket, missing=missing, deep=name == "deepened")
        print(name, flush=True)
    for name, seed in [("reference", 3671), ("repeat", 15731), ("changed", 85411)]:
        counting_scan(root / name, root / ("noisy-" + name), seed)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    generate(cli.parse_args().root)
