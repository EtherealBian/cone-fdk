"""Make two traceable scans of the measured Helsinki pine cone from all eight verified archives."""

import argparse
import configparser
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from common import Geometry
from fetch_pine_cone import META, ROOT, SOURCE, STEP, need, source
from PIL import Image
from slab_replay import stable

RULES = "4fd5f98f528e467532cefa8c498786e6e28fd662b2e09b4a92a8db77b4d69218"
MEMBER = re.compile(r"20201118_pine_cone_(\d{4})\.tif")
MAX_TIFF = 32 << 20


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def publish(path, doc):
    path.write_bytes((json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("utf8"))


def geometry(raw):
    p = configparser.ConfigParser(interpolation=None, strict=True)
    p.read_string(raw.decode("ascii"))
    need(
        p.get("GENERAL", "ProjectName") == "20201118_pine_cone_"
        and p.get("GEOMETRY", "GeometryType") == "Cone"
        and p.get("GEOMETRY", "DistanceUnit") == "mm"
        and p.get("DETECTOR", "PixelSizeUnit") == "mm"
        and p.get("DETECTOR", "Binning") == "1x1",
        "unrecognized acquisition",
    )
    vals = tuple(
        p.getfloat(*x)
        for x in [
            ("ACQUISITION", "AngleFirst"),
            ("ACQUISITION", "AngleInterval"),
            ("ACQUISITION", "AngleLast"),
            ("DETECTOR", "PixelSize"),
            ("GEOMETRY", "DistanceSourceOrigin"),
            ("GEOMETRY", "DistanceSourceDetector"),
        ]
    )
    need(
        p.getint("ACQUISITION", "NumberImages") == 721
        and vals == (0, 0.5, 360, 0.05, 230.66, 553.74),
        "wrong angle, detector pitch or source distances",
    )
    return vals[3:]


def frame(blob):
    with Image.open(io.BytesIO(blob)) as img:
        w, h = img.size
        need(
            img.format == "TIFF"
            and img.n_frames == 1
            and 128 <= w <= 4096
            and 128 <= h <= 4096
            and w * h <= 8 << 20
            and not (w % 8 or h % 8),
            "unsupported TIFF topology",
        )
        a = np.asarray(img)
        need(
            a.shape == (h, w)
            and a.dtype.kind in "ui"
            and a.dtype.itemsize in (2, 4)
            and 0 <= int(a.min()) <= int(a.max()) <= 65535,
            "unsupported detector counts",
        )
        return a.copy()


def intensity(a, independent=False):
    h, w = a.shape
    need(
        a.ndim == 2 and a.dtype.kind in "ui" and h >= 128 and w >= 128 and not (h % 8 or w % 8),
        "detector array",
    )
    if independent:
        # Integer block summation down each axis, independent of reshape/mean layout.
        air = float(np.sum(a[:128, :128], dtype=np.uint64)) / 16384
        block = (
            np.add.reduceat(
                np.add.reduceat(a, np.arange(0, w, 8), axis=1, dtype=np.uint64),
                np.arange(0, h, 8),
                axis=0,
                dtype=np.uint64,
            ).astype("f8")
            / 64
        )
    else:
        air = float(a[:128, :128].mean(dtype="f8"))
        block = a.reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3), dtype="f8")
    need(air > 0 and np.isfinite(air), "nonpositive per-view air reference")
    v = (-np.log(np.maximum(block, 0.5) / air)).astype("<f4")
    need(np.isfinite(v).all(), "nonfinite projection")
    return v, air


def full_file(f, expected):
    m, h = hashlib.md5(), hashlib.sha256()
    n = 0
    while b := f.read(STEP):
        m.update(b)
        h.update(b)
        n += len(b)
    need(
        n == expected["bytes"] and m.hexdigest() == expected["md5"],
        "source archive length or publisher MD5",
    )
    f.seek(0)
    return h.hexdigest()


def index(zips):
    rows = {}
    for name, z in zips.items():
        for m in z.infolist():
            match = MEMBER.fullmatch(m.filename)
            need(
                match
                and m.compress_type in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED)
                and 0 < m.file_size <= MAX_TIFF
                and 0 < m.compress_size <= MAX_TIFF
                and not (m.flag_bits & 1),
                "unexpected ZIP member, size or compression",
            )
            k = int(match[1])
            need(1 <= k <= 721 and k not in rows, "duplicate or out-of-range angle index")
            rows[k] = (name, z, m)
    need(set(rows) == set(range(1, 722)), "missing original angle")
    return rows


def member(z, m):
    with z.open(m) as f:
        b = f.read(m.file_size + 1)
    need(len(b) == m.file_size, "TIFF member length mismatch")
    return b


def prepare(folder, dest):
    need(os.name == "nt", "Windows acquisition guard required; use the Spark entry point on Linux")
    return prepare_verified(folder, dest, lambda path, _pin, _rec: stable(path), os.rename)


def prepare_verified(folder, dest, open_input, publish_directory, platform_sources=None):
    """Shared reconstruction preprocessing; the entry point must provide immutable input streams."""
    folder, dest = folder.resolve(), dest.absolute()
    need(
        dest == dest.resolve(strict=False) and not os.path.lexists(dest)
        and dest.parent.is_dir()
        and not (folder.is_relative_to(dest) or dest.is_relative_to(folder)),
        "fresh separate scan destination required",
    )
    need(
        sha(ROOT / "fixtures/pine-cone-qualification-protocol-v1.json") == RULES,
        "changed pine protocol",
    )
    orig = source()
    record = folder / "source-receipt.json"
    with record.open("rb") as f:
        rb = f.read(65537)
    need(len(rb) <= 65536, "large source receipt")
    receipt = json.loads(rb)
    with (
        ExitStack() as stack,
        tempfile.TemporaryDirectory(dir=dest.parent, prefix=".pine-scan-") as td,
    ):
        stage = Path(td) / "scan"
        stage.mkdir()
        zs = {}
        digests = {}
        for name, x in orig["files"].items():
            f = stack.enter_context(open_input(folder / name, receipt["full_file_sha256"][name], x))
            digests[name] = full_file(f, x)
            if name == META:
                need(
                    digests[name] == orig["acquisition_metadata_sha256"],
                    "changed original metadata",
                )
                pitch, R, D = geometry(f.read())
            else:
                zs[name] = stack.enter_context(zipfile.ZipFile(f))
        need(
            receipt == dict(source_manifest_sha256=SOURCE, full_file_sha256=digests),
            "different download receipt",
        )
        rows = index(zs)
        frames = {}
        shape = None
        air_range = [float("inf"), -float("inf")]
        full = stage / "pine720.f32"
        sparse = stage / "pine90.f32"
        with full.open("xb") as out:
            for k in range(1, 722):
                name, z, m = rows[k]
                b = member(z, m)
                hexdigest = hashlib.sha256(b).hexdigest()
                frames[m.filename] = dict(
                    archive=name, sha256=hexdigest, crc32=f"{m.CRC:08x}", used=(k != 721)
                )
                if k == 721:
                    continue
                a = frame(b)
                v, air = intensity(a)
                del a
                if shape is None:
                    shape = v.shape
                need(v.shape == shape, "inconsistent detector shape")
                air_range = [min(air_range[0], air), max(air_range[1], air)]
                out.write(v.tobytes())
                if k % 100 == 0:
                    print(f"checked {k}/720 unique measured angles", flush=True)
            out.flush()
            os.fsync(out.fileno())
        h, w = shape
        with full.open("rb") as dense, sparse.open("xb") as out:
            for k in range(1, 721, 8):
                _, z, m = rows[k]
                b = member(z, m)
                need(
                    hashlib.sha256(b).hexdigest() == frames[m.filename]["sha256"],
                    "changed sparse source",
                )
                v, _ = intensity(frame(b), independent=True)
                buf = v.tobytes()
                dense.seek((k - 1) * h * w * 4)
                need(
                    v.shape == shape and dense.read(h * w * 4) == buf,
                    "independent sparse preprocessing differs",
                )
                out.write(buf)
            out.flush()
            os.fsync(out.fileno())
        scans = {}
        for n, p in [(90, sparse), (720, full)]:
            need(p.stat().st_size == 4 * n * h * w, "invalid array length")
            g = Geometry(n, h, w, pitch * 8, pitch * 8, R, D)
            geom = p.with_suffix(".geom")
            g.save(geom)
            scans[p.stem] = dict(
                views=n,
                shape_vu=[h, w],
                geom_sha256=sha(geom),
                projections_sha256=sha(p),
                projection_bytes=p.stat().st_size,
            )
        publish(stage / "frames.json", frames)
        report = dict(
            format="CFDK-pine-cone-preprocessing-v1",
            source_manifest_sha256=SOURCE,
            protocol_sha256=RULES,
            original_file_sha256=digests,
            frame_manifest_sha256=sha(stage / "frames.json"),
            used_original_frames=720,
            duplicate_360_excluded=True,
            independently_rechecked_sparse_views=90,
            air_mean_extrema=air_range,
            scans=scans,
            script_sha256=sha(Path(__file__)),
            source_receipt_sha256=hashlib.sha256(rb).hexdigest(),
        )
        if platform_sources is not None:
            report["platform_sources"] = platform_sources
        publish(stage / "scan.json", report)
        publish_directory(stage, dest)
    print("PASS: complete checked originals, 720 distinct angles and independent 90-view subset")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("originals", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    prepare(a.originals, a.destination)
