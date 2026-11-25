"""Inspect a native CFDKROI1 pack or extract one region; Python standard library only."""

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

PREFIX = struct.Struct("<8sIfI")
BOX = struct.Struct("<6I")


def read_index(stream):
    """Validate the header, every extent and exact file length before exposing payloads."""
    stream.seek(0)

    def read(record):
        data = stream.read(record.size)
        if len(data) != record.size:
            raise ValueError("truncated ROI pack header")
        return record.unpack(data)

    magic, n, pitch, count = read(PREFIX)
    if magic != b"CFDKROI1" or not 2 <= n <= 4096 or not 1 <= count <= 256:
        raise ValueError("invalid ROI pack format, parent grid or region count")
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("invalid ROI pack voxel spacing")
    offset = PREFIX.size + count * BOX.size
    entries = []
    for _ in range(count):
        box = read(BOX)
        origin, shape = box[:3], box[3:]
        if any(s == 0 or i + s > n for i, s in zip(origin, shape)):
            raise ValueError("invalid ROI pack extent")
        size = 4 * math.prod(shape)
        entries.append(
            dict(
                origin_xyz=list(origin),
                shape_xyz=list(shape),
                offset_bytes=offset,
                size_bytes=size,
            )
        )
        offset += size
    if stream.seek(0, os.SEEK_END) != offset:
        raise ValueError("ROI pack length does not match its index")
    return dict(
        format="CFDKROI1", n=n, voxel_spacing_mm=pitch, regions=entries, output_bytes=offset
    )


def extract(path, index, dest):
    if dest.exists():
        raise FileExistsError(f"choose a new extraction directory: {dest}")
    with path.open("rb") as src:
        meta = read_index(src)
        if not 0 <= index < len(meta["regions"]):
            raise ValueError("invalid region index (zero-based)")
        entry = meta["regions"][index]
        src.seek(entry["offset_bytes"])
        with TemporaryDirectory(dir=dest.parent) as tmp:
            stage = Path(tmp) / "selected"
            stage.mkdir()
            remaining = entry["size_bytes"]
            digest = hashlib.sha256()
            with (stage / "volume.f32").open("xb") as output:
                while remaining:
                    chunk = src.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("truncated region payload")
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            layout = dict(
                n=meta["n"],
                voxel_spacing_mm=meta["voxel_spacing_mm"],
                roi_origin_xyz=entry["origin_xyz"],
                output_shape_xyz=entry["shape_xyz"],
                output_bytes=entry["size_bytes"],
                region_index=index,
                sha256=digest.hexdigest(),
            )
            (stage / "layout.json").write_text(
                json.dumps(layout, indent=2) + "\n", encoding="utf8"
            )
            # Windows rename does not replace an existing destination directory.
            stage.rename(dest)
    return layout


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for cmd in ("list", "extract"):
        q = sub.add_parser(cmd)
        q.add_argument("pack", type=Path)
        if cmd == "extract":
            q.add_argument("index", type=int, help="zero-based box number in the list")
            q.add_argument("out", type=Path, help="new directory, parent must exist")
    a = p.parse_args()
    if a.action == "list":
        with a.pack.open("rb") as f:
            result = read_index(f)
    else:
        result = extract(a.pack, a.index, a.out)
    print(json.dumps(result, indent=2))
