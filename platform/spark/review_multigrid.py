"""Authenticate a CFDKMG01 pack; optionally export one region with a scanner-local NRRD header. Stdlib only."""

import argparse, hashlib, json, math, os, re, stat, struct, tempfile
from pathlib import Path


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def fp32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def wire(v):
    return tuple(struct.unpack("<Iff6I", v))


def parse(f):
    size = os.fstat(f.fileno()).st_size
    need(12 <= size <= (64 << 30) + 1024, "pack size")
    head = f.read(12)
    need(len(head) == 12, "short header")
    magic, count = struct.unpack("<8sI", head)
    need(magic == b"CFDKMG01" and 1 <= count <= 16, "pack magic/count")
    raw = f.read(36 * count)
    need(len(raw) == 36 * count, "short grid table")
    off = 12 + len(raw)
    items = []
    for off0 in range(0, len(raw), 36):
        n, fov, pitch, x, y, z, nx, ny, nz = wire(raw[off0 : off0 + 36])
        need(
            2 <= n <= 4096
            and math.isfinite(fov)
            and math.isfinite(pitch)
            and pitch > 0
            and fp32(fov / n) == pitch,
            "bad grid",
        )
        need(min(nx, ny, nz) > 0 and x + nx <= n and y + ny <= n and z + nz <= n, "invalid box")
        s = 4 * nx * ny * nz
        origin = [(i - (n - 1) / 2) * pitch for i in (x, y, z)]
        items.append(
            dict(
                n=n,
                fov_mm=fov,
                pitch_mm=pitch,
                origin_xyz=[x, y, z],
                shape_xyz=[nx, ny, nz],
                first_voxel_center_mm=origin,
                offset_bytes=off,
                size_bytes=s,
            )
        )
        off += s
    need(off == size, "missing or trailing payload")
    return head + raw, items


def nrrd(r):
    nx, ny, nz = r["shape_xyz"]
    p = r["pitch_mm"]
    o = ",".join(format(v, ".17g") for v in r["first_voxel_center_mm"])
    return (
        f'NRRD0005\n# Coordinates are scanner-local; no patient orientation or calibration is implied.\ntype: float\ndimension: 3\nspace dimension: 3\nsizes: {nx} {ny} {nz}\nspace directions: ({p},0,0) (0,{p},0) (0,0,{p})\nspace origin: ({o})\nspace units: "mm" "mm" "mm"\ncenters: cell cell cell\nendian: little\nencoding: raw\n\n'
    ).encode("ascii")


def read(f, region=None, dest=None):
    head, items = parse(f)
    need((region is None) == (dest is None), "export needs a selected region and sink")
    need(
        region is None or type(region) is int and 0 <= region < len(items), "invalid region index"
    )
    h = hashlib.sha256(head)
    for i, r in enumerate(items):
        size = r["size_bytes"]
        cell = hashlib.sha256()
        need(f.tell() == r["offset_bytes"], "wrong offset")
        if i == region:
            dest.write(nrrd(r))
        while size:
            v = f.read(min(1 << 20, size))
            need(v, "truncated payload")
            h.update(v)
            cell.update(v)
            size -= len(v)
            if i == region:
                dest.write(v)
        r["sha256"] = cell.hexdigest()
    need(not f.read(1), "trailing payload")
    return dict(format="CFDKMG01", pack_sha256=h.hexdigest(), regions=items)


def verify(path, pin, region=None, out=None):
    need(
        isinstance(pin, str) and re.fullmatch("[0-9a-f]{64}", pin),
        "external pack SHA-256 required",
    )
    need((region is None) == (out is None), "export requires both options")
    tmp = None
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)), "rb") as f:
            need(stat.S_ISREG(os.fstat(f.fileno()).st_mode), "regular pack required")
            if out is None:
                doc = read(f)
            else:
                out = Path(out).absolute()
                need(
                    ".." not in out.parts
                    and out.parent.resolve() == out.parent
                    and not os.path.lexists(out),
                    "fresh output in real directory required",
                )
                with tempfile.NamedTemporaryFile(
                    dir=out.parent, prefix=".grid-nrrd-", delete=False
                ) as sink:
                    tmp = Path(sink.name)
                    doc = read(f, region, sink)
                    sink.flush()
                    os.fsync(sink.fileno())
            need(doc["pack_sha256"] == pin, "pack digest mismatch")
            if tmp is not None:
                os.link(tmp, out)
                if os.name == "posix":
                    fd = os.open(out.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
        return doc
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pack", type=Path)
    p.add_argument("sha256")
    p.add_argument("--region", type=int)
    p.add_argument("--nrrd", type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.pack, a.sha256, a.region, a.nrrd), indent=2))
