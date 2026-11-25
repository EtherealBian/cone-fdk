"""Known cylindrical graphs and semantic tampering of the public offline review."""

import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from review_directory_masks import label, review

DATA = HERE.parents[1] / "reports/gb10"
RECEIPT = DATA / "directory-edge-qualification.json"
PACK = DATA / "directory-edge-masks.zlib"
PIN = "39b156760dc467967d9372701843b392fd07c9347d35b5509d17d8da0723aa31"
sha = lambda b: hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize(
    ("h", "w", "mask", "expected", "regions"),
    [
        (1, 5, [1, 0, 2, 0, 1], [1, 0, -1, 0, 1], 1),  # seam joins the two end columns
        (
            3,
            3,
            [1, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 5, 0, 7, 0, 0],
            3,
        ),  # no diagonals, no axial wrap
        (1, 6, [1, 2, 1, 0, 0, 0], [1, -1, 3, 0, 0, 0], 2),  # unknown does not bridge
        (
            2,
            4,
            [0, 0, 1, 0, 1, 1, 1, 1],
            [0, 0, 3, 0, 3, 3, 3, 3],
            1,
        ),  # minimum row-major ID, not a dense index
    ],
)
def test_known_graphs(h, w, mask, expected, regions):
    assert label(bytes(mask), h, w) == (
        sha(struct.pack("<" + "i" * len(expected), *expected)),
        regions,
        mask.count(2),
        mask.count(1),
    )


def test_committed_classifications():
    assert review(RECEIPT, PIN, PACK) == dict(
        scenario_grids={"repeat": 9, "noisy-change": 9, "invalid-edge-control": 4},
        cells_checked=324720,
        scope="classified synthetic masks only",
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("wrong-receipt", "receipt digest"),
        ("tampered-pack", "mask package"),
        ("trailing", "mask stream"),
        ("truncated", "mask stream"),
        ("expanding", "mask stream"),
        ("changed-cell", "pair"),
        ("wrong-id", "pair"),
        ("wrong-code", "mask code"),
        ("wrong-order", "scenario grid"),
        ("missing-row", "scenario table"),
        ("invented-case", "case set"),
    ],
)
def test_negative_replay(tmp_path, change, reason):
    q = json.loads(RECEIPT.read_bytes())
    pack = PACK.read_bytes()
    raw = bytearray(zlib.decompress(pack))
    if change == "trailing":
        pack += b"\x00"
    elif change == "truncated":
        pack = pack[:-1]
    elif change == "expanding":
        pack = zlib.compress(raw + bytes(1 << 20))
    elif change in ("changed-cell", "wrong-code"):
        raw[0] = 1 if change == "changed-cell" else 3
        pack = zlib.compress(raw)
    elif change == "wrong-id":
        q["cases"]["repeat"]["periodic_label_sha256"][0] = "0" * 64
    elif change == "wrong-order":
        q["cases"]["repeat"]["pairs"].reverse()
    elif change == "missing-row":
        q["cases"]["repeat"]["unknown_cells"].pop()
    elif change == "invented-case":
        q["cases"]["fourth"] = {}
    q["offline_masks"]["sha256"] = sha(pack)
    if change == "tampered-pack":
        pack += b"changed"
    data = json.dumps(q).encode()
    rp = tmp_path / "receipt"
    pp = tmp_path / "pack"
    rp.write_bytes(data)
    pp.write_bytes(pack)
    with pytest.raises(ValueError, match=reason):
        review(rp, "0" * 64 if change == "wrong-receipt" else sha(data), pp)
