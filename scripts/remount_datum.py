"""Four-bead fixture localization. Coordinates in mm, R @ material + t = scanner."""

from itertools import permutations

import numpy as np
from scipy import ndimage


class DatumRejected(ValueError):
    """A valid acquisition did not establish a unique, usable material datum."""


def read_calibration(cfg):
    if cfg["format"] != "CFDK-remount-fixture-v1":
        raise ValueError("unknown fixture format")
    points = np.asarray(cfg["beads_material_xyz_mm"], dtype="f8")
    radius, level, limit, fov = [
        float(cfg[k])
        for k in ("bead_radius_mm", "bead_level_per_mm", "fit_limit_mm", "locator_fov_mm")
    ]
    n = cfg["locator_n"]
    if (
        type(n) is not int
        or not 32 <= n <= 256
        or not all(np.isfinite([radius, level, limit, fov]))
        or min(radius, level, limit, fov) <= 0
        or radius * n / fov < 2.5
        or limit >= radius / 2
    ):
        raise ValueError("invalid locator sampling or fixture parameters")
    if points.shape != (4, 3) or not np.isfinite(points).all():
        raise ValueError("four finite calibrated XYZ positions required")
    if np.linalg.svd(points - points.mean(0), compute_uv=False)[-1] <= 4 * limit:
        raise ValueError("fixture lacks noncoplanar baseline")
    return points, radius, level, limit, n, fov


def scout_slab(n, fov, radius, geom):
    """Z interval that can contain a fully visible bead in an evenly spaced circular scan.

    At least one view has source-to-center axial projection distance <= R. Beyond
    the isocenter's detector-height limit a center cannot pass the all-views gate.
    Include the sphere and two extra voxels; reject islands that touch a cut face.
    """
    if geom.a < 2:
        raise ValueError("at least two views required")
    voxel = fov / n
    max_z = (geom.v - 1) * 0.5 * geom.dv * geom.R / geom.D + radius + 2 * voxel
    lo = max(0, int(np.floor((n - 1) / 2 - max_z / voxel)))
    return lo, n - 2 * lo


def candidates(volume, fov, level, radius, *, z0=0):
    """Threshold a centered scout slab; return positions in the *parent* scanner frame."""
    nz, n, nx = volume.shape
    if n != nx or nz + 2 * z0 != n or z0 < 0 or not np.isfinite(volume).all():
        raise DatumRejected("invalid_scout")
    vox = fov / n
    nominal = 4 * np.pi / 3 * (radius / vox) ** 3
    labels, count = ndimage.label(volume >= level, structure=np.ones((3, 3, 3), bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ids = np.flatnonzero((sizes[1:] >= 0.55 * nominal) & (sizes[1:] <= 1.45 * nominal)) + 1
    if len(ids) != 4:
        raise DatumRejected(f"candidate_count:{len(ids)}")
    slices = ndimage.find_objects(labels)
    records = []
    for idx in ids:
        box = slices[idx - 1]
        if any(s.start == 0 or s.stop == end for s, end in zip(box, volume.shape, strict=True)):
            raise DatumRejected("bead_touches_scout_edge")
        mask = labels[box] == idx
        points = (np.argwhere(mask) + [s.start for s in box])[:, ::-1]
        # Weight by excess above threshold. Attenuation values, accumulation and pose are FP64.
        weights = np.maximum(volume[box][mask].astype("f8") - level, 0)
        if weights.sum() <= 0:
            raise DatumRejected("zero_bead_contrast")
        xyz = (points + [0, 0, z0] - (n - 1) / 2) * vox
        eig = np.linalg.eigvalsh(np.cov(xyz, rowvar=False, bias=True))
        if eig[0] <= 0 or eig[-1] / eig[0] > 1.8:
            raise DatumRejected("non_spherical_candidate")
        records.append(
            dict(
                scanner_xyz_mm=np.average(xyz, axis=0, weights=weights).tolist(),
                voxel_count=int(sizes[idx]),
                shape_variance_ratio=float(eig[-1] / eig[0]),
            )
        )
    return records


def register(template, observed, limit):
    """Unlabeled correspondence: test all 24 proper rigid fits, never a reflection."""
    x = np.asarray(template, dtype="f8")
    y = np.asarray(observed, dtype="f8")
    if (
        x.shape != (4, 3)
        or y.shape != (4, 3)
        or not np.isfinite([x, y]).all()
        or not np.isfinite(limit)
        or limit <= 0
    ):
        raise ValueError("invalid point sets or gate")
    xc = x - x.mean(0)
    if np.linalg.svd(xc, compute_uv=False)[-1] <= 4 * limit:
        raise DatumRejected("degenerate_fixture")
    fits = []
    for p in permutations(range(4)):
        yp = y[list(p)]
        u, _, vt = np.linalg.svd(xc.T @ (yp - yp.mean(0)))
        d = np.eye(3)
        d[-1, -1] = np.linalg.det(vt.T @ u.T)
        rot = vt.T @ d @ u.T
        t = yp.mean(0) - rot @ x.mean(0)
        residual = np.linalg.norm(x @ rot.T + t - yp, axis=1)
        fits.append((float(max(residual)), p, rot, t, residual))
    fits.sort(key=lambda f: f[0])
    best, p, r, t, res = fits[0]
    if best > limit:
        raise DatumRejected("fixture_residual")
    if fits[1][0] <= 2 * limit:
        raise DatumRejected("ambiguous_fixture")
    return dict(
        rotation=r.tolist(),
        translation_mm=t.tolist(),
        candidate_for_template=list(p),
        residuals_mm=res.tolist(),
        max_residual_mm=best,
        next_permutation_max_residual_mm=fits[1][0],
    )


def panel_encloses(g, center, support):
    """Conservative projected bounding box for every acquired view; include bilinear margins."""
    angles = g.angles()
    c = np.cos(angles)
    s = np.sin(angles)
    zero = np.zeros_like(c)
    u = np.stack([c, s, zero], axis=-1)
    normal = np.stack([s, -c, zero], axis=-1)
    z = np.stack([zero, zero, zero + 1], axis=-1)
    denom = g.R - normal @ center - support(normal)
    return bool(
        np.all(denom > 0)
        and np.all((np.abs(u @ center) + support(u)) * g.D < (g.u - 3) * g.du * 0.5 * denom)
        and np.all((abs(center[2]) + support(z)) * g.D < (g.v - 3) * g.dv * 0.5 * denom)
    )
