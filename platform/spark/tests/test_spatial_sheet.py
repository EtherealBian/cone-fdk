"""The independent data oracle works in indexed pixels, not visually similar colors."""

import hashlib
import io
import itertools
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import qualify_spatial_sheet as oracle  # noqa: E402
import spatial_sheet as s  # noqa: E402


@pytest.mark.parametrize("n", [1, 4, 9])
def test_all_legal_index_pairs_and_independent_palette(n):
    pairs = [(f, u) for f in range(n + 1) for u in range(n + 1 - f)]
    buf = bytes(f * 10 + u for f, u in pairs)
    b = s.png(buf, len(buf), 1, n)
    im = Image.open(io.BytesIO(b))
    assert im.mode == "P" and im.tobytes() == buf and len(im.getpalette()) == 300
    for k, (f, u) in enumerate(pairs):
        color = im.convert("RGB").getpixel((k, 0))
        want = (
            (52, 61, 83)
            if u == n
            else (108, 80, 140)
            if u and f
            else (146, 163, 187)
            if u
            else (181, 61, 54)
            if f == n
            else (217, 161, 37)
            if f
            else (223, 232, 230)
        )
        assert color == want
    pos = 8
    tags = []
    while pos < len(b):
        size = struct.unpack_from(">I", b, pos)[0]
        tag = b[pos + 4 : pos + 8]
        tags.append(tag)
        assert struct.unpack_from(">I", b, pos + 8 + size)[0] == zlib.crc32(
            b[pos + 4 : pos + 8 + size]
        )
        pos += size + 12
    assert pos == len(b) and tags == [b"IHDR", b"PLTE", b"IDAT", b"IEND"]


@pytest.mark.parametrize("h,w", [(1, 1), (1, 3), (2, 5), (3, 2), (4, 7)])
def test_exhaustive_valid_crops(h, w):
    src = bytes(range(h * w))
    for y, x in itertools.product(range(h), range(w)):
        for rh, rw in itertools.product(range(1, h - y + 1), range(1, w + 1)):
            rows = [list(src[row * w : (row + 1) * w]) * 2 for row in range(y, y + rh)]
            assert s.crop(src, h, w, (y, x, rh, rw)) == bytes(
                itertools.chain.from_iterable(row[x : x + rw] for row in rows)
            )
    y, x, rh, rw = s.indices(h, w, None)
    assert len(s.crop(src, h, w, (y, x, rh, rw))) == rh * rw


@pytest.mark.parametrize(
    "roi",
    [
        (-1, 0, 1, 1),
        (0, -1, 1, 1),
        (3, 0, 1, 1),
        (0, 4, 1, 1),
        (2, 0, 2, 1),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 0, 1, 5),
        (True, 0, 1, 1),
    ],
)
def test_no_axial_wrap_or_repeat(roi):
    with pytest.raises(ValueError):
        s.indices(3, 4, roi)


@pytest.mark.parametrize(
    "raw,w,h,n",
    [
        (b"\x22", 1, 1, 3),
        (b"\x64", 1, 1, 9),
        (b"", 1, 1, 9),
        (b"\x00", 0, 1, 1),
        (b"\x00", 1, 1, 0),
        (b"\x00", 1, 1, 10),
    ],
)
def test_bad_indices_never_serialized(raw, w, h, n):
    with pytest.raises(ValueError):
        s.png(raw, w, h, n)


def test_patch_reviewer_refuses_before_reading_bundle(tmp_path):
    for n in ["spatial_sheet.py", "offline_campaign.py", "threshold_atlas.py"]:
        shutil.copyfile(HERE / n, tmp_path / n)
    with (tmp_path / "threshold_atlas.py").open("ab") as f:
        f.write(b"\n# test changed source\n")
    p = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(tmp_path / "spatial_sheet.py"),
            str(tmp_path / "missing.cfdk"),
            "0" * 64,
            "test",
        ],
        capture_output=True,
    )
    assert p.returncode != 0 and not p.stdout and b"unrecognized trusted atlas" in p.stderr


@pytest.mark.parametrize(
    "bad",
    [b"<script/>", b'<image href="https://invalid.example/"/>', b"<a/>", b'<path onclick="x"/>'],
)
def test_inspection_rejects_unqualified_markup(bad):
    data = b'<svg xmlns="http://www.w3.org/2000/svg">' + bad + b"</svg>"
    with pytest.raises(ValueError):
        oracle.inspect(data)


def test_same_palette_is_not_same_measurement():
    a = s.png(bytes([10, 20, 30]), 3, 1, 4)
    b = s.png(bytes([20, 30, 10]), 3, 1, 4)
    ia = Image.open(io.BytesIO(a))
    ib = Image.open(io.BytesIO(b))
    assert (
        ia.convert("RGB").tobytes() == ib.convert("RGB").tobytes()
        and ia.tobytes() != ib.tobytes()
        and hashlib.sha256(a).digest() != hashlib.sha256(b).digest()
    )
