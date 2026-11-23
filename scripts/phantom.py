"""Analytic ray intersections, no voxel forward projector or ASTRA dependency."""

import argparse

from common import DATA, Geometry, np

# Finite solid cylinder (axis Z) minus two through-holes; a sphere embedded off-plane.
CYL = [
    (0, 0, 10.5, -8, 8, 0.04),
    (3.2, -2.4, 2.4, -8, 8, -0.04),
    (-4.5, 3.1, 1.5, -8, 8, -0.04),
]
SPHERE = (1.7, 4.1, 3.4, 1.6, 0.035)


def generate(path, views=180, v=128, u=160):
    path.parent.mkdir(parents=True, exist_ok=True)
    g = Geometry(views, v, u, 0.48, 0.48, 120.0, 205.0)
    g.save(path.with_suffix(".geom"))
    xs = (np.arange(u) - (u - 1) / 2) * g.du
    zs = (np.arange(v)[:, None] - (v - 1) / 2) * g.dv
    norm = np.sqrt(g.D**2 + xs**2 + zs**2)
    with path.with_suffix(".f32").open("wb") as out:
        for th in g.angles():
            s, c = np.sin(th), np.cos(th)
            sx, sy = g.R * s, -g.R * c
            dx = (-g.D * s + c * xs) / norm
            dy = (g.D * c + s * xs) / norm
            dz = zs / norm
            acc = np.zeros_like(norm)
            for cx, cy, r, lo, hi, mu in CYL:
                A = dx * dx + dy * dy
                B = (sx - cx) * dx + (sy - cy) * dy
                C = (sx - cx) ** 2 + (sy - cy) ** 2 - r * r
                disc = B * B - A * C
                h = np.sqrt(np.maximum(0, disc))
                ta = (-B - h) / A
                tb = (-B + h) / A
                with np.errstate(divide="ignore"):
                    z1 = lo / dz
                    z2 = hi / dz
                acc += mu * np.where(
                    disc > 0,
                    np.maximum(
                        0,
                        np.minimum(tb, np.maximum(z1, z2)) - np.maximum(ta, np.minimum(z1, z2)),
                    ),
                    0,
                )
            cx, cy, cz, r, mu = SPHERE
            B = (sx - cx) * dx + (sy - cy) * dy - cz * dz
            C = (sx - cx) ** 2 + (sy - cy) ** 2 + cz * cz - r * r
            acc += 2 * mu * np.sqrt(np.maximum(0, B * B - C))
            acc.astype("<f4").tofile(out)
    return g


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--views", type=int, default=180)
    args = a.parse_args()
    generate(DATA / "phantom", args.views)
