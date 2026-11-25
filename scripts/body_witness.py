"""Veto a fixture pose against calibrated workpiece inserts. Never refine the pose here."""

import hashlib
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from remount_datum import panel_encloses
from scipy import ndimage


@dataclass(frozen=True)
class Witness:
    points: np.ndarray
    radius: float
    level: float
    half: float
    limit: float


def calibration(doc, fixture, bead_radius, bead_level, pitch):
    if doc["format"] != "CFDK-body-witness-v1":
        raise ValueError("unknown body witness format")
    pts = np.asarray(doc["material_xyz_mm"], dtype="f8")
    r, level, half, limit = [
        float(doc[k])
        for k in ("radius_mm", "level_per_mm", "search_half_mm", "residual_limit_mm")
    ]
    if (
        pts.shape != (3, 3)
        or not np.isfinite(pts).all()
        or not np.isfinite([r, level, half, limit]).all()
        or not 0 < level < bead_level
        or r < 2.5 * pitch
        or not 0 < limit < r / 2
        or half <= r + limit + pitch
    ):
        raise ValueError("invalid body witness sampling or gate")
    # Search boxes cannot overlap at any rotation; exclude fixture beads even at box corners.
    reach = np.sqrt(3) * half
    if (
        any(np.linalg.norm(a - b) <= 2 * reach for a, b in combinations(pts, 2))
        or np.min(np.linalg.norm(pts[:, None] - fixture, axis=2)) <= reach + bead_radius
    ):
        raise ValueError("body witness search windows are not isolated")
    # Smallest triangle altitude, invariant to point order, not just a duplicate-point check.
    altitude = np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])) / max(
        np.linalg.norm(a - b) for a, b in combinations(pts, 2)
    )
    if altitude <= max(2 * r, 10 * limit):
        raise ValueError("body witnesses lack a non-collinear baseline")
    return Witness(pts, r, level, half, limit)


def inspect(volume, z0, fov, geom, fit, cfg):
    """Rounded-in windows on the CUDA scout; returned coordinates are parent scanner XYZ."""
    nz, n, nx = volume.shape
    if nx != n or nz + 2 * z0 != n or z0 < 0:
        raise ValueError("unexpected scout slab geometry")
    pitch = fov / n
    origin = (n - 1) * 0.5
    observations = []
    for i, point in enumerate(cfg.points):
        expected = np.asarray(fit["rotation"]) @ point + fit["translation_mm"]
        record = dict(
            id=i + 1,
            predicted_scanner_xyz_mm=expected.tolist(),
            observed_scanner_xyz_mm=None,
            residual_mm=None,
            status="window_outside_scout",
        )
        observations.append(record)
        lo = np.ceil((expected - cfg.half) / pitch + origin).astype(int)
        hi = np.floor((expected + cfg.half) / pitch + origin).astype(int) + 1
        if np.any(lo < [0, 0, z0]) or np.any(hi > [n, n, z0 + nz]):
            continue
        # Require visibility of the actual sampled box, not just its predicted center.
        center = ((lo + hi - 1) * 0.5 - origin) * pitch
        half = (hi - lo - 1) * pitch * 0.5
        record["parent_roi_xyz_nxyz"] = np.r_[lo, hi - lo].tolist()
        if not panel_encloses(geom, center, lambda v, half=half: abs(v) @ half):
            record["status"] = "window_outside_detector"
            continue
        box = volume[lo[2] - z0 : hi[2] - z0, lo[1] : hi[1], lo[0] : hi[0]]
        record["window_sha256"] = hashlib.sha256(box.tobytes(order="C")).hexdigest()
        record.update(locate(box, lo, origin, pitch, expected, cfg))
    return observations


def locate(box, lo, origin, pitch, expected, cfg):
    if not np.isfinite(box).all():
        return dict(status="invalid_window")
    values = box.astype("f8")  # Compare against the calibrated FP64 level without FP32 rounding.
    mask = values >= cfg.level
    _, count = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    info = dict(component_count=int(count))
    if count != 1:
        return dict(**info, status="missing" if count == 0 else "ambiguous")
    ijk = np.argwhere(mask)
    if np.any(ijk.min(0) == 0) or np.any(ijk.max(0) == np.array(box.shape) - 1):
        return dict(**info, status="clipped")
    nominal = 4 * np.pi / 3 * (cfg.radius / pitch) ** 3
    info["voxel_count"] = len(ijk)
    if not 0.55 * nominal <= len(ijk) <= 1.45 * nominal:
        return dict(**info, status="wrong_volume")
    xyz = (ijk[:, ::-1] + lo - origin) * pitch
    eig = np.linalg.eigvalsh(np.cov(xyz, rowvar=False, bias=True))
    if eig[0] <= 0 or eig[-1] > 1.8 * eig[0]:
        return dict(**info, status="wrong_shape")
    weights = values[mask] - cfg.level
    if weights.sum() <= 0:
        return dict(**info, status="zero_contrast")
    obs = np.average(xyz, axis=0, weights=weights)
    err = float(np.linalg.norm(obs - expected))
    return dict(
        **info,
        observed_scanner_xyz_mm=obs.tolist(),
        residual_mm=err,
        status="pass" if err <= cfg.limit else "displaced",
    )
