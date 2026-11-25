"""CPU-only centroid audit for the three observed inserts in a compact pivot-case evidence pack."""

import argparse
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path

NAMES = [
    "manifest.json",
    "fixture.json",
    "body-witness.json",
    *(f"insert-{i}.f32" for i in (1, 2, 3)),
]


def audit(path):
    with zipfile.ZipFile(path) as z:
        if (
            sorted(z.namelist()) != sorted(NAMES)
            or sum(i.file_size for i in z.infolist()) > 2 * 1024**2
        ):
            raise ValueError("unexpected evidence pack")
        raw = {name: z.read(name) for name in NAMES}
    job = json.loads(raw["manifest.json"])
    if job["format"] != "CFDK-remount-witness-job-v1":
        raise ValueError("wrong job format")
    for name in ["fixture.json", "body-witness.json"]:
        if hashlib.sha256(raw[name]).hexdigest() != job["inputs_sha256"][name]:
            raise ValueError("calibration digest mismatch")
    config = json.loads(raw["body-witness.json"])
    fixture = json.loads(raw["fixture.json"])
    n = fixture["locator_n"]
    pitch = fixture["locator_fov_mm"] / n
    scan = job["scans"]["inspected"]
    fit = scan["material_to_scanner"]
    result = []
    for i, (rec, point) in enumerate(
        zip(scan["body_witnesses"], config["material_xyz_mm"], strict=True), 1
    ):
        payload = raw[f"insert-{i}.f32"]
        if rec["id"] != i or hashlib.sha256(payload).hexdigest() != rec["window_sha256"]:
            raise ValueError("window digest or ID mismatch")
        x, y, z, w, h, d = rec["parent_roi_xyz_nxyz"]
        if min(w, h, d) <= 0 or len(payload) != 4 * w * h * d:
            raise ValueError("invalid window dimensions")
        mass = 0.0
        sums = [0.0] * 3
        count = 0
        for j, (val,) in enumerate(struct.iter_unpack("<f", payload)):
            if val >= config["level_per_mm"]:
                weight = val - config["level_per_mm"]
                xyz = [x + j % w, y + j // w % h, z + j // (w * h)]
                count += 1
                mass += weight
                for k in range(3):
                    sums[k] += ((xyz[k] - (n - 1) / 2) * pitch) * weight
        if count != rec["voxel_count"] or mass <= 0:
            raise ValueError("window has no verifiable centroid")
        obs = [s / mass for s in sums]
        pred = [
            sum(a * b for a, b in zip(row, point, strict=True)) + t
            for row, t in zip(fit["rotation"], fit["translation_mm"], strict=True)
        ]
        err = math.dist(pred, obs)
        status = "pass" if err <= config["residual_limit_mm"] else "displaced"
        if (
            not math.dist(pred, rec["predicted_scanner_xyz_mm"]) <= 1e-10
            or not math.dist(obs, rec["observed_scanner_xyz_mm"]) <= 1e-10
            or not math.isclose(err, rec["residual_mm"], abs_tol=1e-10, rel_tol=0)
            or status != rec["status"]
        ):
            raise ValueError("centroid or classification mismatch")
        result.append(dict(id=i, residual_mm=err, status=status))
    if len(result) != 3:
        raise ValueError("three observed inserts required")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pack", type=Path)
    print(json.dumps(audit(p.parse_args().pack), indent=2))
