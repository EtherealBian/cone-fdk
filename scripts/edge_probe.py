"""Recompute one cell's independent level pairs from an unpacked edge review (standard library only)."""

import argparse
import hashlib
import itertools
import json
import math
import struct
from pathlib import Path

STATUS = ("measured", "empty", "clipped", "multiple", "ambiguous", "nonfinite", "undersampled")
CODES = ("always_below", "sensitive", "always_flagged", "unknown")


def probe(job, row, column):
    doc = json.loads((job / "manifest.json").read_bytes())
    if doc["format"] != "CFDK-edge-review-v1" or doc["class_codes"] != list(CODES):
        raise ValueError("unknown review format")
    h, w = doc["shape_ha"]
    levels = doc["levels_per_mm"]
    gate = doc["loss_gate_mm"]
    if not (
        0 <= row < h
        and 0 <= column < w
        and 2 <= len(levels) <= 16
        and levels == sorted(set(levels))
        and all(math.isfinite(v) and v > 0 for v in [gate, *levels])
    ):
        raise ValueError("bad cell coordinates or scenario metadata")

    def record(name, fmt):
        n = struct.calcsize(fmt)
        with (job / name).open("rb") as f:
            if (
                hashlib.file_digest(f, "sha256").hexdigest() != doc["outputs_sha256"][name]
                or f.seek(0, 2) != n * h * w
            ):
                raise ValueError("review map digest or size mismatch: " + name)
            f.seek((row * w + column) * n)
            return struct.unpack(fmt, f.read(n))

    samples = {}
    for role in ("reference", "inspected"):
        samples[role] = []
        for i, level in enumerate(levels):
            inner, outer, t, status, res = record(f"{role}-{i:02}.wall", "<dddII")
            if (
                res
                or status >= len(STATUS)
                or not (
                    all(math.isnan(x) for x in (inner, outer, t))
                    if status
                    else (
                        all(math.isfinite(x) for x in (inner, outer, t))
                        and 0 < inner < outer
                        and t > 0
                        and abs(outer - inner - t) <= 16 * math.ulp(1.0) * outer
                    )
                )
            ):
                raise ValueError("invalid wall sample")
            samples[role].append(
                dict(
                    level_per_mm=level, status=STATUS[status], thickness_mm=None if status else t
                )
            )
    pairs = []
    for a, b in itertools.product(samples["reference"], samples["inspected"]):
        ok = a["status"] == b["status"] == "measured"
        loss = a["thickness_mm"] - b["thickness_mm"] if ok else None
        pairs.append(
            dict(
                reference_level=a["level_per_mm"],
                inspected_level=b["level_per_mm"],
                loss_mm=loss,
                flagged=(loss >= gate if ok else None),
            )
        )
    values = [p["loss_mm"] for p in pairs if p["loss_mm"] is not None]
    unknown = len(values) != len(pairs)
    lo, hi = (None, None) if unknown else (min(values), max(values))
    code = 3 if unknown else 2 if lo >= gate else 1 if hi >= gate else 0
    actual = [record(n, "<d")[0] for n in ["min-loss.f64", "max-loss.f64"]]
    if record("class.u8", "<B")[0] != code or not (
        all(math.isnan(v) for v in actual) if unknown else actual == [lo, hi]
    ):
        raise ValueError("stored envelope disagrees with explicit pair replay")
    return dict(
        row=row,
        column=column,
        loss_gate_mm=gate,
        classification=CODES[code],
        min_loss_mm=lo,
        max_loss_mm=hi,
        samples=samples,
        pairs=pairs,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job", type=Path)
    p.add_argument("--row", required=True, type=int)
    p.add_argument("--column", required=True, type=int)
    print(json.dumps(probe(**vars(p.parse_args())), indent=2, allow_nan=False))
