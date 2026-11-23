"""Build two 90-view, 8x8-binned offline fixtures from measured Helsinki projections.
Supports full verified local ZIPs or range reads of individual members (ZIP CRC + pinned
per-frame SHA256 in the release manifest). Does not download/extract entire archives by default.
"""

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import pathlib
import struct
import subprocess
import time
import zipfile
import zlib

from common import DATA, ROOT, Geometry, np
from fetch import MD5, SIZES, digest
from PIL import Image

CACHE = DATA / "downloads" / "walnut_frames"
CACHE.mkdir(parents=True, exist_ok=True)


def remote(i, a, b):
    name = f"20201111_walnut_projections_{i}.zip"
    url = f"https://zenodo.org/api/records/6986012/files/{name}/content?r={a}-{b}"
    with __import__("tempfile").TemporaryDirectory(dir=CACHE) as tmp:
        h = pathlib.Path(tmp) / "header"
        out = pathlib.Path(tmp) / "part"
        for j in range(5):
            p = subprocess.run(
                [
                    "curl.exe",
                    "-fsSL",
                    "--max-time",
                    "170",
                    "-D",
                    str(h),
                    "-r",
                    f"{a}-{b}",
                    "-o",
                    str(out),
                    url,
                ],
                capture_output=True,
            )
            if (
                not p.returncode
                and f"content-range: bytes {a}-{b}/{SIZES[i - 1]}" in h.read_text().lower()
                and out.stat().st_size == b - a + 1
            ):
                return out.read_bytes()
            time.sleep(3 * (j + 1))
        raise RuntimeError(f"failed range {i} {a}:{b}: {p.stderr!r}")


def indices(i):
    size = SIZES[i - 1]
    tail = remote(i, size - 65557, size - 1)
    p = tail.rfind(b"PK\x05\x06")
    _, _, _, _, count, sz, off, _ = struct.unpack_from("<4s4H2IH", tail, p)
    p = off - (size - len(tail))
    records = []
    for j in range(count):
        fields = struct.unpack_from("<4s6H3I5H2I", tail, p)
        assert fields[0] == b"PK\x01\x02" and fields[4] == 8
        nl, el, cl = fields[10:13]
        name = tail[p + 46 : p + 46 + nl].decode()
        records.append((name, fields[-1], fields[8], fields[7]))
        p += 46 + nl + el + cl
    return records


def bin_views(raw, binning):
    a = np.asarray(Image.open(io.BytesIO(raw)))
    h, w = a.shape
    k = binning
    assert not (w % k or h % k)
    air = np.mean(a[:128, :128], dtype="f8")
    result = []
    for shift in (0, -5):
        img = np.roll(a, shift, axis=1) if shift else a
        avg = img.reshape(h // k, k, w // k, k).mean(axis=(1, 3))
        result.append((-np.log(np.maximum(avg, 0.5) / air)).astype("<f4"))
    return result


def prepare(stride=8, binning=8, jobs=8):
    if 720 % stride or binning not in (4, 8, 16):
        raise ValueError("stride must divide 720; binning 4,8,16")
    prefix = ROOT / "fixtures" / f"walnut-s{stride}-b{binning}"
    prefix.mkdir(parents=True, exist_ok=True)
    expected = prefix / "frames.json"
    pinned = json.loads(expected.read_text()) if expected.exists() else {}
    truth = {}
    processed = {}
    for i in range(1, 9):
        name = f"20201111_walnut_projections_{i}.zip"
        zip_path = DATA / "downloads" / name
        local = zip_path.exists() and zip_path.stat().st_size == SIZES[i - 1]
        if local:
            if digest(zip_path) != MD5[i - 1]:
                raise RuntimeError("corrupt archive " + name)
            with zipfile.ZipFile(zip_path) as z:
                rec = [
                    (e.filename, e.header_offset, e.compress_size, e.CRC) for e in z.infolist()
                ]
        else:
            rec = indices(i)
        chosen = [
            r for r in rec if (int(r[0][-8:-4]) - 1) % stride == 0 and int(r[0][-8:-4]) <= 720
        ]

        def one(r, local=local, zip_path=zip_path, i=i):
            name, off, sz, crc = r
            cache = CACHE / name
            if cache.exists():
                raw = cache.read_bytes()
            elif local:
                with zipfile.ZipFile(zip_path) as z:
                    raw = z.read(name)
            else:
                # Fixed 30-byte header; read up to 2 KiB of name/extra along with payload.
                b = remote(i, off, min(off + 30 + 2048 + sz - 1, SIZES[i - 1] - 1))
                assert b[:4] == b"PK\x03\x04"
                nl, el = struct.unpack_from("<HH", b, 26)
                assert nl + el <= 2048 and b[30 : 30 + nl].decode() == name
                raw = zlib.decompress(b[30 + nl + el : 30 + nl + el + sz], -15)
            if zlib.crc32(raw) != crc:
                raise ValueError("bad ZIP CRC: " + name)
            sha = hashlib.sha256(raw).hexdigest()
            if pinned and pinned.get(name) != sha:
                raise ValueError("SHA256 mismatch " + name)
            if not cache.exists():
                cache.write_bytes(raw)
            return name, sha, bin_views(raw, binning)

        with cf.ThreadPoolExecutor(jobs) as pool:
            for name, sha, pair in pool.map(one, chosen):
                truth[name] = sha
                processed[name] = pair
        print(f"archive {i}/8: {len(processed)} measured views", flush=True)
    assert len(processed) == 720 // stride
    expected.write_text(json.dumps(truth, sort_keys=True, indent=2) + "\n")
    ordered = sorted(processed)
    h, w = processed[ordered[0]][0].shape
    g = Geometry(len(ordered), h, w, 0.05 * binning, 0.05 * binning, 210.66, 553.74)
    for j, name in enumerate(["uncorrected", "corrected"]):
        scan = prefix / name
        g.save(scan.with_suffix(".geom"))
        np.savez_compressed(
            scan.with_suffix(".npz"), views=np.stack([processed[n][j] for n in ordered])
        )
    return prefix


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--bin", type=int, default=8)
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()
    prepare(a.stride, a.bin, a.jobs)
