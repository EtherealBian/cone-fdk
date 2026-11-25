"""Replay one reported cell from the original pair of 32-byte wall records. Standard library only."""

import argparse
import hashlib
import json
import math
import struct
import zipfile
from contextlib import ExitStack
from pathlib import Path

SAMPLE = struct.Struct("<dddII")
STATUS = ("measured", "empty", "clipped", "multiple", "ambiguous", "nonfinite", "undersampled")
MEMBERS = ("grid.txt", "reference.wall", "inspected.wall", "regions.json")


def probe(doc, ref, part, row, column):
    h, w = doc["rows"], doc["columns"]
    if doc["format"] != "CFDK-loss-v1" or not (0 <= row < h <= 4096 and 0 <= column < w <= 4096):
        raise ValueError("unexpected report or cell outside map")

    def read(f):
        if f.seek(0, 2) != h * w * SAMPLE.size:
            raise ValueError("wall map length differs from report")
        f.seek((row * w + column) * SAMPLE.size)
        inner, outer, thickness, status, reserved = SAMPLE.unpack(f.read(SAMPLE.size))
        if reserved or status >= len(STATUS):
            raise ValueError("invalid wall status")
        vals = (inner, outer, thickness)
        if status == 0:
            if not (
                all(map(math.isfinite, vals))
                and 0 < inner < outer
                and thickness > 0
                and abs((outer - inner) - thickness) <= 16 * math.ulp(1.0) * outer
            ):
                raise ValueError("invalid measured record")
        elif not all(map(math.isnan, vals)):
            raise ValueError("rejected cell contains dimensions")
        return status, dict(
            status=STATUS[status],
            inner_mm=None if status else inner,
            outer_mm=None if status else outer,
            thickness_mm=None if status else thickness,
        )

    a, first = read(ref)
    b, second = read(part)
    candidates = [code for y, x, n, code in doc["runs"] if y == row and x <= column < x + n]
    if len(candidates) > 1:
        raise ValueError("overlapping runs at requested cell")
    code = candidates[0] if candidates else 0
    delta = None
    threshold = doc["threshold_mm"]
    if not (math.isfinite(threshold) and threshold > 0):
        raise ValueError("invalid threshold")
    if a or b:
        if code != -2 - len(STATUS) * a - b:
            raise ValueError("unknown reason disagrees with wall records")
        status = "unknown"
    else:
        loss = first["thickness_mm"] - second["thickness_mm"]
        if code < 0 or (code > 0) != (loss >= threshold):
            raise ValueError("classification disagrees with wall records")
        status = "flagged" if code else "below_threshold"
        delta = dict(
            loss_mm=loss,
            inner_growth_mm=second["inner_mm"] - first["inner_mm"],
            outer_recession_mm=first["outer_mm"] - second["outer_mm"],
        )
    peak = False
    if code > 0:
        regs = [r for r in doc["regions"] if r["id"] == code]
        if len(regs) != 1:
            raise ValueError("missing or repeated region ID")
        p = regs[0]["peak"]
        peak = (p["row"], p["column"]) == (row, column)
        if delta["loss_mm"] > p["loss_mm"] or (peak and any(delta[k] != p[k] for k in delta)):
            raise ValueError("reported peak disagrees with replay")
    return dict(
        row=row,
        column=column,
        azimuth_deg=360 * column / w,
        axial_offset_mm=(row - (h - 1) / 2) * doc["axial_pitch_mm"],
        threshold_mm=threshold,
        status=status,
        region_id=code if code > 0 else None,
        is_reported_peak=peak,
        reference=first,
        inspected=second,
        delta=delta,
    )


def write_demo(path, grid, reference, part, report):
    # Fixed metadata: identical source bytes make identical review packs on the qualified toolchain.
    payload = {
        n: p.read_bytes() for n, p in zip(MEMBERS, (grid, reference, part, report), strict=True)
    }
    payload["SHA256.json"] = (
        json.dumps({n: hashlib.sha256(s).hexdigest() for n, s in payload.items()}, indent=2)
        + "\n"
    ).encode()
    with zipfile.ZipFile(path, "w") as z:
        for name, raw in payload.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 0
            info.external_attr = 0x20
            z.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED)


def from_bundle(path, row, column):
    # A convenience for small evidence packs, not the production grid size limit. No extraction.
    with zipfile.ZipFile(path) as z:
        files = z.infolist()
        if (
            sorted(i.filename for i in files) != sorted((*MEMBERS, "SHA256.json"))
            or sum(i.file_size for i in files) > 16 * 1024**2
        ):
            raise ValueError("invalid review pack")
        digests = json.loads(z.read("SHA256.json"))
        for n in MEMBERS:
            with z.open(n) as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != digests[n]:
                    raise ValueError("review pack digest mismatch: " + n)
        with z.open("reference.wall") as ref, z.open("inspected.wall") as part:
            return probe(json.loads(z.read("regions.json")), ref, part, row, column)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    bundled = subs.add_parser("bundle", help="verify and probe a portable qualification sample")
    bundled.add_argument("zip", type=Path)
    raw = subs.add_parser("raw", help="probe your own native report and wall files")
    for name in ("report", "reference", "inspected"):
        raw.add_argument(name, type=Path)
    for p in (bundled, raw):
        p.add_argument("--row", type=int, required=True)
        p.add_argument("--column", type=int, required=True)
    a = parser.parse_args()
    if a.command == "bundle":
        result = from_bundle(a.zip, a.row, a.column)
    else:
        with ExitStack() as stack:
            result = probe(
                json.loads(a.report.read_bytes()),
                stack.enter_context(a.reference.open("rb")),
                stack.enter_context(a.inspected.open("rb")),
                a.row,
                a.column,
            )
    print(json.dumps(result, indent=2, allow_nan=False))
