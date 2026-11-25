"""Finite tube with two independent material-removal operations. Analytic cone-ray intervals."""

import numpy as np
from common import Geometry

RO, RI, MU = 9.4, 6.1, 0.04
DIMS = (360, 41, 121)
STEPS = (4.5, 0.05, 0.2)
# Flat: azimuth, depth, lower/upper Z. Relief: azimuth, cutter-center radius,
# cutter radius, lower/upper Z. The relief connects to the bore but not the OD.
FLAT = (0.0, 0.85, -2.7, -0.9)
RELIEF = (134.0, 5.05, 1.8, 0.5, 2.7)
CASES = {
    "reference": (None, None),
    "outer-flat": (FLAT, None),
    "dual-surface": (FLAT, RELIEF),
    "subthreshold": ((51.0, 0.12, -1.1, 2.7), None),
}


def pencil_circle(sx, sy, dx, dy, cx, cy, r):
    ax, ay = cx - sx, cy - sy
    t = ax * dx + ay * dy
    half = np.sqrt(np.maximum(0, r * r - (ax * dy - ay * dx) ** 2))
    return t - half, t + half


def chord_in_slab(slope, interval, lo, hi):
    a, b = lo / slope, hi / slope
    left, r = interval
    return np.maximum(0, np.minimum(r, np.maximum(a, b)) - np.maximum(left, np.minimum(a, b)))


def project(scan, flat, relief):
    g = Geometry(360, 80, 320, 0.14, 0.24, 120.0, 205.0)
    g.save(scan.with_suffix(".geom"))
    u = (np.arange(g.u) - (g.u - 1) / 2) * g.du
    v = (np.arange(g.v)[:, None] - (g.v - 1) / 2) * g.dv
    with scan.with_suffix(".f32").open("wb") as f:
        for theta in g.angles():
            s, c = np.sin(theta), np.cos(theta)
            sx, sy = g.R * s, -g.R * c
            view = np.zeros((g.v, g.u))
            for off in (-1 / 3, 0, 1 / 3):
                x = u + off * g.du
                norm = np.hypot(g.D, x)
                dx, dy = (-g.D * s + c * x) / norm, (g.D * c + s * x) / norm
                slope = v / norm

                outer = pencil_circle(sx, sy, dx, dy, 0, 0, RO)
                bore = pencil_circle(sx, sy, dx, dy, 0, 0, RI)
                material = chord_in_slab(slope, outer, -7, 7) - chord_in_slab(slope, bore, -7, 7)
                if flat is not None:
                    angle, depth, z0, z1 = flat
                    nx, ny = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
                    b = RO - depth - nx * sx - ny * sy
                    m = nx * dx + ny * dy
                    # Clip the outer chord against n.P >= RO-depth. Handle parallel rays explicitly.
                    nonparallel = abs(m) > 1e-14
                    cross = np.divide(b, m, out=np.zeros_like(m), where=nonparallel)
                    left = np.where(nonparallel & (m > 0), np.maximum(outer[0], cross), outer[0])
                    r = np.where(nonparallel & (m < 0), np.minimum(outer[1], cross), outer[1])
                    material -= np.where(
                        nonparallel | (b <= 0), chord_in_slab(slope, (left, r), z0, z1), 0
                    )
                if relief is not None:
                    angle, e, rc, z0, z1 = relief
                    th = np.deg2rad(angle)
                    cut = pencil_circle(sx, sy, dx, dy, e * np.cos(th), e * np.sin(th), rc)
                    overlap = (np.maximum(cut[0], bore[0]), np.minimum(cut[1], bore[1]))
                    material -= chord_in_slab(slope, cut, z0, z1) - chord_in_slab(
                        slope, overlap, z0, z1
                    )
                view += material * (MU / 3) * np.hypot(1, slope)
            view.astype("<f4").tofile(f)


def design(flat, relief, seam=0):
    # Independent radial intersections, no reused projection arrays, wall records or native labels.
    w, h, _ = DIMS
    z = ((np.arange(h) - (h - 1) / 2) * STEPS[2])[:, None]
    phi = np.deg2rad(seam + 360 * np.arange(w) / w)
    dent = np.zeros((h, w))
    growth = dent.copy()
    if flat is not None:
        angle, depth, z0, z1 = flat
        c = np.cos(phi - np.deg2rad(angle))
        dent = ((z >= z0) & (z <= z1)) * np.maximum(0, RO - (RO - depth) / np.maximum(c, 1e-8))
    if relief is not None:
        angle, e, r, z0, z1 = relief
        q = phi - np.deg2rad(angle)
        disc = r * r - (e * np.sin(q)) ** 2
        growth = (
            ((z >= z0) & (z <= z1))
            * np.maximum(0, e * np.cos(q) + np.sqrt(np.maximum(0, disc)) - RI)
            * (disc >= 0)
        )
    return dent, growth


PHOTONS = 250_000
NOISE_SEEDS = {
    "noisy-reference": ("reference", 3671),
    "noisy-clean": ("reference", 15731),
    "noisy-dual": ("dual-surface", 85411),
}


def counting_scan(source, out, seed):
    """Independent counting noise on native detector bins before FDK (not noise on a map)."""
    g = Geometry.read(source.with_suffix(".geom"))
    g.save(out.with_suffix(".geom"))
    rng = np.random.Generator(np.random.PCG64(seed))
    with source.with_suffix(".f32").open("rb") as f, out.with_suffix(".f32").open("wb") as to:
        for _ in range(g.a):
            att = np.fromfile(f, "<f4", g.u * g.v).astype("f8")
            counts = rng.poisson(PHOTONS * np.exp(-att))
            (-np.log(np.maximum(1, counts) / PHOTONS)).astype("<f4").tofile(to)
