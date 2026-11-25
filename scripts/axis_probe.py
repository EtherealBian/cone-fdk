"""CPU-only second opinion from coarse.grid, coarse.wall and axis.json. Requires only NumPy."""

import argparse
import json
from pathlib import Path

import numpy as np

RECORD = np.dtype(
    [
        ("inner", "<f8"),
        ("outer", "<f8"),
        ("thickness", "<f8"),
        ("status", "<u4"),
        ("reserved", "<u4"),
    ]
)


def angle(u, v):
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))))


def review(job):
    fields = (job / "coarse.grid").read_text(encoding="utf-8-sig").split()
    if len(fields) != 16 or fields[0] != "CFDKCYL1":
        raise ValueError("invalid grid")
    w, h, d = map(int, fields[1:4])
    center, axis, seam, steps = np.array(fields[4:], float).reshape(4, 3)
    if not (32 <= w <= 4096 and w % 2 == 0 and 3 <= h <= 4096 and 3 <= d <= 4096):
        raise ValueError("invalid dimensions")
    if not (
        np.isfinite([center, axis, seam, steps]).all()
        and (steps > 0).all()
        and abs(np.linalg.norm(axis) - 1) <= 1e-6
        and abs(np.linalg.norm(seam) - 1) <= 1e-6
        and abs(axis @ seam) <= 1e-6
    ):
        raise ValueError("invalid coarse frame or sampling pitch")
    axis /= np.linalg.norm(axis)
    seam -= np.dot(axis, seam) * axis
    seam /= np.linalg.norm(seam)
    frame = np.stack([seam, np.cross(axis, seam), axis])
    z = (np.arange(h) - 0.5 * (h - 1)) * steps[2]
    fit = json.loads((job / "axis.json").read_bytes())
    m = fit["measurement"]
    if not (
        fit["format"] == "CFDK-axis-v1"
        and fit["status"] == "measured"
        and m
        and (h, w) == (fit["rows"], fit["columns"])
    ):
        raise ValueError("only an accepted matching axis can be checked")
    t = 2 * np.pi * np.arange(w) / w
    rays = np.stack([np.cos(t), np.sin(t)], axis=1)
    reported = np.asarray(m["axis_unit"])
    origin = np.asarray(m["origin_xyz_mm"])
    radius = m["outer_radius_mm"]
    if not (
        abs(np.linalg.norm(reported) - 1) < 1e-10 and np.isfinite(origin).all() and radius > 0
    ):
        raise ValueError("invalid measurement")
    # A reversed line or an origin shifted ALONG it gives the same distances. Check the
    # representation as well as the geometry before accepting a point-distance replay.
    max_radius = steps[0] + (d - 1) * steps[1]
    tol = 64 * np.finfo(float).eps * max(1, max_radius, np.linalg.norm(center))
    expected_seam = seam - (seam @ reported) * reported
    if not (
        angle(axis, reported) <= 20 + 1e-10
        and abs((origin - center) @ axis) <= tol
        and np.linalg.norm(
            np.array(m["seam_unit"]) - expected_seam / np.linalg.norm(expected_seam)
        )
        < 1e-10
        and fit["valid_samples"] == w * h
        and np.isfinite(radius)
        and z[-1] - z[0] >= 0.25 * radius - tol
        and abs(fit["span_mm"] - (z[-1] - z[0])) <= tol
        and np.isfinite(fit["max_residual_mm"])
        and fit["max_residual_mm"] > 0
    ):
        raise ValueError("axis frame, coverage or span disagrees with the grid")
    file = job / "coarse.wall"
    if file.stat().st_size != h * w * RECORD.itemsize:
        raise ValueError("wrong wall length")
    offsets = []
    tot = peak = 0.0
    worst = None
    with file.open("rb") as src:
        for j in range(h):
            r = np.fromfile(src, RECORD, w)
            outer = r["outer"]
            if (
                r["status"].any()
                or r["reserved"].any()
                or not (
                    np.isfinite(outer).all()
                    and (outer > r["inner"]).all()
                    and (r["inner"] > 0).all()
                    and (r["inner"] >= steps[0] * (1 - 16 * np.finfo(float).eps)).all()
                    and (outer <= max_radius * (1 + 16 * np.finfo(float).eps)).all()
                    and (r["thickness"] > 0).all()
                    and (
                        abs(r["thickness"] - (outer - r["inner"]))
                        <= 16 * np.finfo(float).eps * outer
                    ).all()
                )
            ):
                raise ValueError("invalid wall record")
            xy = outer[:, None] * rays
            x, y = xy.T
            # Dense least squares, independent of diametric pairing and Fourier moments.
            a, b, c, lx, ly = np.linalg.lstsq(
                np.c_[x * x, 2 * x * y, y * y, x, y], np.ones(w), rcond=None
            )[0]
            offsets.append(np.linalg.solve([[a, b], [b, c]], [-lx / 2, -ly / 2]))
            xyz = center + xy @ frame[:2] + z[j] * axis
            distances = np.linalg.norm(np.cross(xyz - origin, reported), axis=1)
            err = abs(distances - radius)
            i = int(err.argmax())
            tot += float(distances.sum())
            if worst is None or err[i] > peak:
                peak = float(err[i])
                worst = [j, i]
    intercept, slope = np.linalg.lstsq(np.c_[np.ones(h), z], offsets, rcond=None)[0]
    center2 = center + intercept @ frame[:2]
    u = axis + slope @ frame[:2]
    u /= np.linalg.norm(u)
    radius_delta = abs(tot / (h * w) - radius)
    peak_delta = abs(peak - fit["peak_residual_mm"])
    if not (
        max(radius_delta, peak_delta) <= 1e-10 * max(1, radius)
        and peak <= fit["max_residual_mm"] + 1e-12
    ):
        raise ValueError("reported distances disagree with original outer points")
    return dict(
        samples=w * h,
        alternative_axis_unit=u.tolist(),
        alternative_origin_xyz_mm=center2.tolist(),
        alternative_angle_delta_deg=angle(u, reported),
        alternative_origin_delta_mm=float(np.linalg.norm(center2 - origin)),
        recomputed_peak_mm=peak,
        worst_row_column=worst,
        radius_replay_delta_mm=radius_delta,
        peak_replay_delta_mm=peak_delta,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job", type=Path)
    a = p.parse_args()
    print(json.dumps(review(a.job), indent=2, allow_nan=False))
