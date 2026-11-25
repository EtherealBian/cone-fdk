"""Use published CUDA data without requiring a GPU to exercise sparse authenticated disclosure."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import periodic_parcel
import periodic_roi as roi
import pytest

ROOT = Path(__file__).resolve().parents[1]
PARCEL = "961648e72ce25d1e3d36b1f8a576af3e63750a2b809ca4e160a183e075ba2395"
ZIP = "d8c833a2de5e3dea4101fa9605f95834bd8ea7d1618ff5d421ea485be7142078"
BOX = [19, 355, 2, 10]


@pytest.fixture
def bundle(tmp_path):
    original = ROOT / "reports/periodic-seam-demo.zip"
    assert roi.digest(original.read_bytes()) == ZIP
    b = tmp_path / "source"
    with zipfile.ZipFile(original) as z:
        z.extractall(b)
    return b


@pytest.fixture
def indexed(bundle, tmp_path):
    idx = tmp_path / "index.json"
    return bundle, idx, roi.index(bundle, idx, PARCEL, 4)


def query(row, box=BOX, out=None):
    bundle, idx, ip = row
    return roi.inspect(bundle, idx, PARCEL, ip, box, dest=out)


def verify(out, ip, box=BOX):
    return roi.inspect(out, out / "index.json", PARCEL, ip, box, detached=True)


def test_detached_reviewer_does_not_have_parent_or_science_stack(indexed, tmp_path):
    b, idx, ip = indexed
    out = tmp_path / "show"
    got = query(indexed, out=out)
    assert got["wraps_azimuth"] and got["observed_cells"] == got["requested_cells"] == 20
    assert (
        sum(got["observed_cells_by_global_id"].values()) == 20
        and len(got["observed_cells_by_global_id"]) == 1
    )
    assert got["authenticated_rows_half_open"] == [16, 24]
    assert got["authenticated_payload_bytes"] == 14400
    for name in ("manifest.json", "mask.u8", "labels.i32"):
        (b / name).unlink()
    b.rmdir()
    idx.unlink()
    p = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(ROOT / "scripts/periodic_roi.py"),
            "verify",
            str(out),
            "--parcel-pin",
            PARCEL,
            "--index-pin",
            ip,
            "--box",
            *map(str, BOX),
        ],
        check=True,
        capture_output=True,
    )
    assert json.loads(p.stdout) == got
    with pytest.raises(ValueError, match="requested ROI"):
        verify(out, ip, [19, 354, 2, 10])


@pytest.mark.parametrize(
    "box",
    [
        [-1, 0, 1, 1],
        [40, 0, 2, 1],
        [0, -1, 1, 1],
        [0, 360, 1, 1],
        [0, 1, 1, 361],
        [0, 1, 0, 1],
        [0, 1, 2, 0],
        [False, 0, 1, 1],
    ],
)
def test_no_fabricated_axis_or_area(indexed, box):
    with pytest.raises(ValueError, match="invalid ROI"):
        query(indexed, box)


def test_one_turn_and_single_bottom_row(indexed):
    v = query(indexed, [40, 359, 1, 360])
    assert (
        v["authenticated_rows_half_open"] == [40, 41]
        and v["requested_cells"] == 360
        and v["wraps_azimuth"]
    )
    assert query(indexed, [40, 359, 1, 1])["wraps_azimuth"] is False


@pytest.mark.parametrize("key", ["parcel", "index"])
def test_wrong_root(indexed, key):
    b, i, ip = indexed
    with pytest.raises(ValueError, match="SHA-256"):
        roi.inspect(
            b, i, "0" * 64 if key == "parcel" else PARCEL, "0" * 64 if key == "index" else ip, BOX
        )


@pytest.mark.parametrize("damage", ["swapped", "bit", "missing", "extra", "long", "summary"])
def test_refuse_incomplete_or_altered_detached(indexed, tmp_path, damage):
    out = tmp_path / "cut"
    query(indexed, out=out)
    p = out / roi.member(4, "labels.i32")
    if damage == "swapped":
        p.write_bytes((out / roi.member(5, "labels.i32")).read_bytes())
    elif damage == "bit":
        data = bytearray(p.read_bytes())
        data[-1] ^= 1
        p.write_bytes(data)
    elif damage == "missing":
        p.unlink()
    elif damage == "extra":
        (out / "comment").write_bytes(b"annotation")
    elif damage == "long":
        with p.open("ab") as f:
            f.write(b"z")
    else:
        (out / "receipt.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        verify(out, indexed[2])


def test_touch_catches_damage_but_does_not_certify_other_rows(indexed, tmp_path):
    b, i, ip = indexed
    before = query(indexed)
    with (b / "mask.u8").open("r+b") as f:
        f.write(b"\x02")
    assert query(indexed) == before
    with pytest.raises(ValueError, match="invalid touched block"):
        query(indexed, [0, 17, 1, 1], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError):
        periodic_parcel.offline(b, PARCEL, 4)


def test_index_does_not_publish_after_late_mutation(bundle, tmp_path, monkeypatch):
    real = periodic_parcel.offline

    def changed(*args):
        v = real(*args)
        with (bundle / "labels.i32").open("r+b") as f:
            f.write(b"X")
        return v

    monkeypatch.setattr(periodic_parcel, "offline", changed)
    target = tmp_path / "new.json"
    with pytest.raises(ValueError, match="changed after global"):
        roi.index(bundle, target, PARCEL, 4)
    assert not target.exists()


def test_publish_refuses_other_destinations(indexed, tmp_path):
    b, i, ip = indexed
    original = i.read_bytes()
    with pytest.raises(ValueError):
        roi.index(b, i, PARCEL, 4)
    assert i.read_bytes() == original
    with pytest.raises(ValueError):
        query(indexed, out=b / "nested")
    receipt = tmp_path / "receipt"
    receipt.mkdir()
    with pytest.raises(ValueError):
        query(indexed, out=receipt)
    assert not list(receipt.iterdir())


@pytest.mark.parametrize("rows", [0, 42, True])
def test_block_bounds(bundle, tmp_path, rows):
    with pytest.raises(ValueError):
        roi.index(bundle, tmp_path / "bad", PARCEL, rows)


def test_different_block_height_same_crop(indexed, tmp_path):
    b, i, ip = indexed
    i2 = tmp_path / "index-7"
    alt = roi.index(b, i2, PARCEL, 7)
    a = query(indexed)
    v = query((b, i2, alt))
    assert ip != alt
    for k in ("mask_u8_sha256", "labels_i32le_sha256", "observed_cells_by_global_id"):
        assert a[k] == v[k]
    assert a["authenticated_rows_half_open"] != v["authenticated_rows_half_open"]


def test_protocol_bytes_precede_qualification():
    assert (
        roi.digest((ROOT / "fixtures/periodic-roi-receipt-protocol-v1.json").read_bytes())
        == roi.PROTOCOL
    )


@pytest.mark.parametrize("box", [[39, 359, 2, 2], [0, 18, 1, 1]])
def test_reader_measures_only_authenticated_row_bytes(indexed, monkeypatch, box):
    bundle, _, _ = indexed
    handles = {}
    real = Path.open

    class Tracked:
        def __init__(self, f):
            self.f = f
            self.reads = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()

        def seek(self, pos):
            return self.f.seek(pos)

        def read(self, n):
            self.reads.append((self.f.tell(), n))
            return self.f.read(n)

    def opening(p, *args, **kwargs):
        f = real(p, *args, **kwargs)
        if p.parent == bundle and p.name in {"mask.u8", "labels.i32"}:
            handles[p.name] = Tracked(f)
            return handles[p.name]
        return f

    monkeypatch.setattr(Path, "open", opening)
    result = query(indexed, box)
    lo, hi = result["authenticated_rows_half_open"]
    for n, size in (("mask.u8", 1), ("labels.i32", 4)):
        assert handles[n].reads == [
            (y * 360 * size, (min(y + 4, 41) - y) * 360 * size) for y in range(lo, hi, 4)
        ]
    assert (
        sum(count for f in handles.values() for _, count in f.reads)
        == result["authenticated_payload_bytes"]
    )
