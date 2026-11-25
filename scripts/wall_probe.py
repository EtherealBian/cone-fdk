"""Replay one measured or rejected ray from the saved atlas; no GPU or CT rerun needed."""

import argparse
import json
from pathlib import Path

import numpy as np
from wall import STATUS, WALL_DTYPE


def probe(atlas, wall, meta, row, column):
    if meta["format"] != "CFDK-wall-v1-le":
        raise ValueError("not a CFDK wall v1 map")
    h, w = meta["shape_ha"]
    d = meta["shells"]
    level = meta["threshold"]
    r0 = meta["first_radius_mm"]
    dr = meta["radial_step_mm"]
    if not (0 <= row < h and 0 <= column < w):
        raise ValueError("probe is outside the wall map")
    if (
        atlas.stat().st_size != 4 * d * h * w
        or wall.stat().st_size != WALL_DTYPE.itemsize * h * w
    ):
        raise ValueError("payload size disagrees with metadata")
    a = np.memmap(atlas, "<f4", "r", shape=(d, h, w))[:, row, column].astype("f8")
    record = np.memmap(wall, WALL_DTYPE, "r", shape=(h, w))[row, column]
    radius = r0 + np.arange(d) * dr
    mask = a > level
    pairs = np.flatnonzero(np.diff(mask))
    knots = np.flatnonzero(a[1:-1] == level) + 1
    finite = np.isfinite(a).all()
    if not finite:
        status = "nonfinite"
    elif max(a[0], a[-1]) >= level:
        status = "clipped"
    elif any(not (min(a[i - 1], a[i + 1]) < level < max(a[i - 1], a[i + 1])) for i in knots):
        status = "ambiguous"
    elif len(pairs) > 2:
        status = "multiple"
    elif not len(pairs):
        status = "empty"
    elif np.count_nonzero(mask) < 2:
        status = "undersampled"
    else:
        status = "measured"
    if STATUS.index(status) != record["status"] or record["reserved"] != 0:
        raise ValueError("stored status disagrees with input profile")
    # Summarize even rejected multi-segment profiles, but do not infer a wall from their edges.
    crossings = []
    for i in pairs if finite else []:
        frac = (level - a[i]) / (a[i + 1] - a[i])
        crossings.append(
            dict(
                k=int(i),
                direction="enter" if mask[i + 1] else "exit",
                bracket_radius_mm=radius[i : i + 2].tolist(),
                bracket_attenuation=a[i : i + 2].tolist(),
                linear_radius_mm=float(r0 + (i + frac) * dr),
            )
        )
    vals = [float(record[k]) for k in ("inner", "outer", "thickness")]
    result = None
    if status == "measured":
        ri, ro = [v["linear_radius_mm"] for v in crossings]
        if not np.allclose(
            vals, [ri, ro, ro - ri], rtol=0, atol=8 * np.spacing(max(abs(ri), abs(ro)))
        ):
            raise ValueError("stored dimensions disagree with input profile")
        angle = 2 * np.pi * column / w
        axis = np.array(meta["axis_unit"])
        seam = np.array(meta["seam_unit"])
        origin = np.array(meta["center_mm"]) + (row - (h - 1) / 2) * meta["axial_pitch_mm"] * axis
        radial = np.cos(angle) * seam + np.sin(angle) * np.cross(axis, seam)
        result = dict(
            inner_mm=vals[0],
            outer_mm=vals[1],
            thickness_mm=vals[2],
            inner_xyz_mm=(origin + vals[0] * radial).tolist(),
            outer_xyz_mm=(origin + vals[1] * radial).tolist(),
        )
    elif not np.isnan(vals).all():
        raise ValueError("rejected record contains a dimension")
    return dict(
        row=row,
        column=column,
        azimuth_deg=360 * column / w,
        status=status,
        threshold=level,
        first_radius_mm=r0,
        radial_step_mm=dr,
        attenuation=[float(v) if np.isfinite(v) else None for v in a],
        crossings=crossings,
        measurement=result,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("atlas", type=Path)
    p.add_argument("wall", type=Path)
    p.add_argument("metadata", type=Path, help="stdout JSON from cone-fdk wall / scripts/wall.py")
    p.add_argument("--row", required=True, type=int)
    p.add_argument("--column", required=True, type=int)
    a = p.parse_args()
    with a.metadata.open("rb") as f:
        meta = json.load(f)
    print(json.dumps(probe(a.atlas, a.wall, meta, a.row, a.column), indent=2, allow_nan=False))
