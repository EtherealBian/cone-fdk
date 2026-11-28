"""Independent FP64 reference: explicit source/detector geometry and spatial convolution."""

import numpy as np


def backproject_points(projections, g, points):
    """Evaluate FDK at XYZ world points (mm)."""
    points = np.asarray(points, dtype="f8")
    js = np.arange(1 - g.u, g.u)
    h = np.zeros(len(js), "f8")
    h[js == 0] = 0.25
    odd = js % 2 == 1
    h[odd] = -1 / (np.pi * js[odd]) ** 2
    v, u = np.indices((g.v, g.u))
    u = (u - (g.u - 1) / 2) * g.du
    v = (v - (g.v - 1) / 2) * g.dv
    cosine = g.D / np.sqrt(g.D**2 + u * u + v * v)
    out = np.zeros(points.shape[:-1], dtype="f8")
    for a, p in enumerate(projections):
        angle = g.start + g.sign * 2 * np.pi * a / g.a
        # Build a source, a detector plane and its two basis vectors in world coordinates.
        normal = np.array([-np.sin(angle), np.cos(angle), 0])
        source = -g.R * normal
        uc = np.array([np.cos(angle), np.sin(angle), 0])
        vc = np.array([0.0, 0.0, 1.0])
        center = source + g.D * normal
        rays = points - source
        den = rays @ normal
        hit = source + rays * (g.D / den)[..., None] - center
        uf = (hit @ uc) / g.du + (g.u - 1) / 2
        vf = (hit @ vc) / g.dv + (g.v - 1) / 2
        # Direct linear spatial convolution; keep the unpadded detector samples.
        rows = np.stack(
            [np.convolve(row, h, "full")[g.u - 1 : 2 * g.u - 1] for row in p * cosine]
        )
        ui = np.floor(uf).astype(int)
        vi = np.floor(vf).astype(int)
        samples = np.zeros_like(out)
        for dv in (0, 1):
            for du in (0, 1):
                xx = ui + du
                yy = vi + dv
                mask = (xx >= 0) & (xx < g.u) & (yy >= 0) & (yy < g.v)
                weight = (1 - np.abs(uf - xx)) * (1 - np.abs(vf - yy))
                samples[mask] += rows[yy[mask], xx[mask]] * weight[mask]
        out += samples * (g.R / den) ** 2
    return out * (np.pi / g.a) * (g.D / (g.R * g.du))
