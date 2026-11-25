"""Pre-acquisition geometric prediction for the registered neck challenge; not a CT result."""

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
from remount_loss import write_doc

SPEC = Path(__file__).resolve().parents[1] / "fixtures/bridge-study-v1.json"
SPEC_SHA256 = "27d7fb4aa8683def6089ca0e4548e6b68d582de6ec9b55cb28f3857e39f4a322"


def protocol():
    b = SPEC.read_bytes()
    if hashlib.sha256(b).hexdigest() != SPEC_SHA256:
        raise ValueError("bridge protocol changed; use a new study version")
    return json.loads(b)


def loss_field(spec, depth):
    """Outward material-coordinate radial ray to a planar OD cut (no detector model here)."""
    m = spec["measurement"]
    h, w = m["shape_ha"]
    r = spec["tube_mm"]["outer_radius"]
    if depth is None:
        return np.zeros((h, w), dtype="f8")
    if not 0 <= depth < r - spec["tube_mm"]["inner_radius"]:
        raise ValueError("cut intersects the bore")
    z = (np.arange(h) - (h - 1) / 2) * m["axial_step_mm"]
    parts = np.digitize(z, spec["axial_breaks_mm"])
    shoulder = spec["shoulder_depth_mm"]
    d = np.array([0, shoulder, depth, shoulder, 0])[parts, None]
    cos = np.cos(np.deg2rad(360 * np.arange(w) / w - spec["flat_normal_deg"]))
    # d - (R-d)*(sec(phi)-1) preserves exact equality at the aligned ray (cos=1).
    term = np.divide(1 - cos, cos, out=np.zeros(w), where=cos > 0)
    return np.where(cos > 0, np.maximum(0, d - (r - d) * term), 0)


def linked(mask, anchors):
    """Boolean reachability; unmeasured cells must not be quietly coerced to True."""
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("expected an explicitly measured Boolean mask")
    h, w = mask.shape
    a, b = map(tuple, anchors)
    if not all(0 <= y < h and 0 <= x < w for y, x in (a, b)):
        raise ValueError("anchor outside grid")
    if not mask[a] or not mask[b]:
        return False
    seen = {a}
    q = deque([a])
    while q:
        y, x = q.popleft()
        if (y, x) == b:
            return True
        for pt in ((y, (x + 1) % w), (y, (x - 1) % w), (y - 1, x), (y + 1, x)):
            if 0 <= pt[0] < h and mask[pt] and pt not in seen:
                seen.add(pt)
                q.append(pt)
    return False


def predict():
    spec = protocol()
    m = spec["measurement"]
    h, _ = m["shape_ha"]
    rows = {}
    for scan in spec["scans"]:
        field = loss_field(spec, scan["neck_depth_mm"])
        mask = field >= m["loss_gate_mm"]
        rows[scan["id"]] = dict(
            flagged_cells=int(mask.sum()),
            central_row_flagged_cols=int(mask[h // 2].sum()),
            both_anchors_flagged=all(bool(mask[tuple(a)]) for a in m["anchors_row_col"]),
            anchors_connected=linked(mask, m["anchors_row_col"]),
            peak_mm=float(field.max()),
            mask_u8_sha256=hashlib.sha256(mask.astype("u1").tobytes()).hexdigest(),
        )
    return dict(
        format="CFDK-bridge-design-v1",
        scope="Analytic geometry only, no simulated detector acquisitions or physical measurements.",
        study_sha256=SPEC_SHA256,
        cases=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    out = p.parse_args().output
    if out.exists():
        p.exit(1, "choose a fresh output file\n")
    write_doc(out, predict())
