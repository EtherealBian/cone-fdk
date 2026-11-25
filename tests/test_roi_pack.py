"""Native multi-ROI scheduling, archive protocol and failure/rollback gates."""

import hashlib
import io
import os

import numpy as np
import pytest
from common import reconstruct
from export_volume import export_layout, verify_viewer
from phantom import generate
from roi_pack import BOX, PREFIX, extract, read_index

BOXES = [
    (5, 7, 9, 67, 69, 53),
    (11, 13, 15, 37, 35, 17),
    (5, 7, 9, 67, 69, 53),
    (96, 96, 96, 1, 1, 1),
]


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    p = tmp_path_factory.mktemp("multi") / "scan"
    generate(p, 37, 32, 48)
    return p


@pytest.mark.parametrize("budget", [1, 32])
def test_mixed_boxes_and_split(scan, tmp_path, budget):
    specs = [",".join(map(str, b)) for b in BOXES]
    lst = tmp_path / "选区.txt"
    lst.write_bytes(("\ufeff # Example\r\n\r\n" + "\r\n".join(specs)).encode("utf8"))
    pack = tmp_path / "结果.pack"
    m = reconstruct(scan, pack, 97, 7.37, budget, 7, roi_list=lst)
    assert m["redzones_verified"] and m["peak_owned_bytes"] <= budget * 2**20
    assert (m["projection_passes"] == 1) == (budget == 32)
    assert (
        m["projection_bytes_read"]
        == m["projection_passes"] * scan.with_suffix(".f32").stat().st_size
    )
    assert m["filter_batches"] == m["projection_passes"] * 6  # 37=7*5+2
    assert m["launches_bp"] >= len(BOXES) * 6
    with pack.open("rb") as f:
        meta = read_index(f)
        assert meta["regions"] == m["regions"] and meta["output_bytes"] == m["output_bytes"]
        end = 20 + 24 * len(BOXES)
        for i, (b, entry) in enumerate(zip(BOXES, meta["regions"])):
            assert tuple(entry["origin_xyz"] + entry["shape_xyz"]) == b
            assert entry["offset_bytes"] == end
            f.seek(end)
            expected = tmp_path / "single.f32"
            solo = reconstruct(scan, expected, 97, 7.37, budget, 7, roi=specs[i])
            if i == 0 and budget == 1:
                assert solo["slabs"] > 1
            assert expected.read_bytes() == f.read(entry["size_bytes"])
            end += entry["size_bytes"]
        assert end == pack.stat().st_size
    dest = tmp_path / "review"
    layout = extract(pack, 1, dest)
    raw = (dest / "volume.f32").read_bytes()
    assert layout["sha256"] == hashlib.sha256(raw).hexdigest()
    entry = m["regions"][1]
    off = entry["offset_bytes"]
    assert raw == pack.read_bytes()[off : off + entry["size_bytes"]]
    page = export_layout(dest / "volume.f32", dest / "layout.json", dest)
    assert verify_viewer(page, 17, size_xy=(37, 35), z0=15) == 17
    with pytest.raises(FileExistsError):
        extract(pack, 1, dest)
    with pytest.raises(ValueError):
        extract(pack, -1, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "kind", ["empty", "extra", "range", "many", "both", "alias", "hardlink", "plane", "nan"]
)
def test_bad_plan_preserves_output(scan, tmp_path, kind):
    lst = tmp_path / "boxes.txt"
    out = tmp_path / "old.pack"
    one = "1,1,1,9,9,9\n"
    data = {
        "empty": "   \n# no boxes\n",
        "extra": one + "1,2,3,4,5,6,\n",
        "many": one * 257,
        "range": one + "90,1,1,9,9,9",
        "plane": "0,0,0,4000,4000,1",
    }.get(kind, one)
    lst.write_text(data)
    if kind == "alias":
        out = lst
    elif kind == "hardlink":
        os.link(lst, out)
    else:
        out.write_bytes(b"keep prior result")
    before = out.read_bytes()
    if kind == "nan":
        p = tmp_path / "bad"
        p.with_suffix(".geom").write_bytes(scan.with_suffix(".geom").read_bytes())
        raw = np.fromfile(scan.with_suffix(".f32"), "<f4")
        raw[-1] = float("nan")
        raw.tofile(p.with_suffix(".f32"))
        scan = p
    with pytest.raises(RuntimeError):
        reconstruct(
            scan,
            out,
            4096 if kind == "plane" else 97,
            7.37,
            1,
            7,
            roi_list=lst,
            roi=one.strip() if kind == "both" else None,
        )
    assert out.read_bytes() == before and not list(tmp_path.glob("*.partial.*"))


@pytest.mark.parametrize(
    "kind", ["short", "magic", "count", "grid", "nan", "extent", "index", "tail", "long"]
)
def test_reader_rejects_corruption(kind):
    prefix = PREFIX.pack(
        b"badmagic" if kind == "magic" else b"CFDKROI1",
        1 if kind == "grid" else 17,
        float("nan") if kind == "nan" else 0.5,
        0xFFFFFFFF if kind == "count" else 1,
    )
    desc = BOX.pack(17 if kind == "extent" else 0, 1, 2, 1, 1, 1)
    buf = prefix + desc + b"\x00" * 4
    buf = {
        "short": prefix[:7],
        "index": prefix + desc[:3],
        "tail": buf[:-1],
        "long": buf + b"x",
    }.get(kind, buf)
    with pytest.raises(ValueError):
        read_index(io.BytesIO(buf))


@pytest.mark.parametrize("count", [1, 256])
def test_list_limits_and_batch_fallback(scan, tmp_path, count):
    # Every position has a distinct index in a tiny packed atlas. Starting at
    # batch 37 cannot fit the filter allocation in 1 MiB, even for one voxel.
    origin = (39, 41, 43)
    xs, ys, zs = origin
    points = [(xs + i % 8, ys + i // 8 % 4, zs + i // 32, 1, 1, 1) for i in range(count)]
    lst = tmp_path / "points.txt"
    lst.write_text("\n".join(",".join(map(str, p)) for p in points))
    pack = tmp_path / "points.croi"
    m = reconstruct(scan, pack, 97, 7.37, 1, 37, roi_list=lst)
    assert m["batch"] < 37 and m["projection_passes"] == 1 and m["peak_owned_bytes"] <= 2**20
    assert len(m["regions"]) == count
    ref = tmp_path / "box.f32"
    reconstruct(scan, ref, 97, 7.37, 1, m["batch"], roi=f"{xs},{ys},{zs},8,4,8")
    with pack.open("rb") as f:
        meta = read_index(f)
        f.seek(meta["regions"][0]["offset_bytes"])
        assert f.read() == ref.read_bytes()[: 4 * count]
