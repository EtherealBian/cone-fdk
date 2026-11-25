"""Finite extruded annuli at independent 3-D poses. Analytic rays, no CT or wall fit code."""

from pathlib import Path

import numpy as np
from common import Geometry

MU = 0.04
# tilt, heading, outer axis intercept at scanner z=0, outer semiaxes, bore semiaxes,
# bore eccentricity and phase (in the part frame), all dimensions in mm.
CASES = {
    "upright": (0, 0, (0.3, -0.25), (9.6, 9.6), (6.2, 6.2), 1.1, 55),
    "tilted-same-part": (8, 41, (-0.45, 0.55), (9.6, 9.6), (6.2, 6.2), 1.1, -19),
    "tilted-second-part": (12, 152, (0.6, -0.35), (8.7, 8.7), (5.9, 5.9), 0.65, 110),
    "oval-outside": (8, 41, (-0.45, 0.55), (9.6, 9.2), (6.2, 6.2), 1.1, -19),
    "oval-bore": (12, 152, (0.6, -0.35), (8.7, 8.7), (5.9, 5.5), 0.65, 110),
}


NOISY = "tilted-same-part-noisy"
CASES[NOISY] = CASES["tilted-same-part"]
PHOTONS, SEED = 500_000, 71717


def design(case):
    tilt, heading, xy, outer, bore, ecc, phase = case
    a, b, p = np.deg2rad([tilt, heading, phase])
    axis = np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])
    e = np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), -np.sin(a)])
    f = np.cross(axis, e)
    center = np.r_[xy, 0.0]
    offset = ecc * (np.cos(p) * e + np.sin(p) * f)
    return center, axis, e, f, offset


def project(scan, case, *, noisy=False):
    rng = np.random.default_rng(SEED) if noisy else None
    center, axis, e, f, offset = design(case)
    outer, bore = case[3:5]
    g = Geometry(360, 144, 640, 0.07, 0.14, 120.0, 205.0)
    g.save(scan.with_suffix(".geom"))
    u = (np.arange(g.u) - (g.u - 1) / 2) * g.du
    v = (np.arange(g.v)[:, None] - (g.v - 1) / 2) * g.dv
    with scan.with_suffix(".f32").open("wb") as file:
        for angle in g.angles():
            s, c = np.sin(angle), np.cos(angle)
            origin = g.R * np.array([s, -c, 0.0])
            view = np.zeros((g.v, g.u))
            for shift in (-1 / 3, 0, 1 / 3):
                u1 = (u + shift * g.du) / g.D
                dx, dy, dz = -s + c * u1, c + s * u1, v / g.D
                length = np.sqrt(1 + u1 * u1 + dz * dz)

                def component(vec, dx=dx, dy=dy, dz=dz):
                    return dx * vec[0] + dy * vec[1] + dz * vec[2]

                ax = component(axis)
                z0 = (origin - center) @ axis
                # Intersect the forward ray with both cap planes, including parallel rays.
                parallel = abs(ax) < 1e-14
                divisor = np.where(parallel, 1, ax)
                t1, t2 = (-18 - z0) / divisor, (18 - z0) / divisor
                near = np.where(parallel, 0, np.maximum(0, np.minimum(t1, t2)))
                far = np.where(
                    parallel, np.where(abs(z0) <= 18, np.inf, -np.inf), np.maximum(t1, t2)
                )
                for pos, radii, sign in [(center, outer, 1), (center + offset, bore, -1)]:
                    a, b = radii
                    x, y = ((origin - pos) @ e) / a, ((origin - pos) @ f) / b
                    de, df = component(e) / a, component(f) / b
                    q = de * de + df * df
                    mid = -(x * de + y * df) / q
                    half = np.sqrt(np.maximum(0, q - (x * df - y * de) ** 2)) / q
                    view += (
                        (sign * MU / 3)
                        * length
                        * np.maximum(
                            0, np.minimum(far, mid + half) - np.maximum(near, mid - half)
                        )
                    )
            if rng is not None:
                counts = rng.poisson(PHOTONS * np.exp(-view))
                view = -np.log(np.maximum(counts, 1) / PHOTONS)
            view.astype("<f4").tofile(file)


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    for name, case in CASES.items():
        project(root / name, case, noisy=name == NOISY)
        print(name, flush=True)
