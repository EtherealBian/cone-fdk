"""Build a row-block index after global certification, then issue or inspect detached ROI evidence.

Only the index producer needs the complete parcel verifier and its dependencies. `query`, `pack`
and `verify` load Python's standard library only. Rows never wrap; the requested columns may.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import tempfile
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path

PROTOCOL = "a05630e04c348abf03897082c74740d7ac00a4ad3246ab7094873089f58530a0"
INDEX = "CFDK-periodic-row-index-v1"
RECEIPT = "CFDK-periodic-roi-receipt-v1"
PARENT = "CFDK-periodic-label-parcel-v1"
MAX_CELLS = 1 << 20
LIMIT = 2 << 20
PAYLOADS = (("mask.u8", 1, "mask_u8_sha256"), ("labels.i32", 4, "labels_i32le_sha256"))


def need(ok, why):
    if not ok:
        raise ValueError(why)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(v):
    need(isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v), "expected external SHA-256")
    return v


def encode(data):
    return (
        json.dumps(data, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def regular(p):
    need(not p.is_symlink() and p.is_file(), "expected a regular file: " + p.name)
    return p


def bounded(p, maximum, expected=None):
    with regular(p).open("rb") as f:
        size = os.fstat(f.fileno()).st_size
        need(0 < size <= maximum, "outside file size limit: " + p.name)
        b = f.read(size)
        need(len(b) == size and not f.read(1), "changed file length")
    if expected is not None:
        need(digest(b) == pin(expected), "SHA-256 mismatch: " + p.name)
    return b


def integer(v, lo, hi):
    return type(v) is int and lo <= v <= hi


def grid(doc, band):
    need(doc["format"] == PARENT, "not a certified label parcel")
    h, w = doc["shape_ha"]
    need(integer(h, 1, 4096) and integer(w, 3, 4096), "parcel shape")
    need(integer(band, 1, h) and band * w <= MAX_CELLS, "index block size")
    return h, w


def atomic_file(path, data):
    # Publish by hard link so a concurrent writer cannot overwrite an existing index.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            tmp = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def index(bundle, dest, parcel_pin, block_rows):
    from periodic_parcel import offline

    bundle, dest = Path(bundle).resolve(), Path(dest).resolve()
    need(
        not dest.exists() and not dest.is_relative_to(bundle),
        "index must be fresh and outside source",
    )
    raw = bounded(bundle / "manifest.json", 1 << 20, pin(parcel_pin))
    doc = json.loads(raw)
    h, w = grid(doc, block_rows)
    proof = offline(bundle, parcel_pin, block_rows)["certificate"]
    sums = [hashlib.sha256(), hashlib.sha256()]
    blocks = []
    with ExitStack() as stack:
        fs = [stack.enter_context(regular(bundle / n).open("rb")) for n, _, _ in PAYLOADS]
        for y in range(0, h, block_rows):
            row = []
            for f, total, (_, size, _) in zip(fs, sums, PAYLOADS, strict=True):
                count = min(block_rows, h - y) * w * size
                b = f.read(count)
                need(len(b) == count, "short source block")
                total.update(b)
                row.append(digest(b))
            blocks.append(row)
        need(all(not f.read(1) for f in fs), "long source payload")
    for total, (_, _, key) in zip(sums, PAYLOADS, strict=True):
        need(
            total.hexdigest() == pin(doc["certificate"][key]) == proof[key],
            "source changed after global certification",
        )
    data = encode(
        dict(
            format=INDEX,
            protocol_sha256=PROTOCOL,
            parcel_manifest_sha256=parcel_pin,
            shape_ha=[h, w],
            block_rows=block_rows,
            blocks=blocks,
        )
    )
    need(len(data) <= LIMIT, "oversized index")
    atomic_file(dest, data)
    return digest(data)


def metadata(bundle, idx, parcel_pin, index_pin):
    manifest = bounded(bundle / "manifest.json", 1 << 20, pin(parcel_pin))
    raw = bounded(idx, LIMIT, pin(index_pin))
    doc, info = json.loads(manifest), json.loads(raw)
    need(
        info["format"] == INDEX
        and info["protocol_sha256"] == PROTOCOL
        and info["parcel_manifest_sha256"] == parcel_pin,
        "index provenance",
    )
    band = info["block_rows"]
    h, w = grid(doc, band)
    rows = info["blocks"]
    need(
        encode(info["shape_ha"]) == encode([h, w])
        and isinstance(rows, list)
        and len(rows) == (h + band - 1) // band,
        "index coverage",
    )
    for r in rows:
        need(isinstance(r, list) and len(r) == 2, "index block record")
        for v in r:
            pin(v)
    return doc, info, manifest, raw


def area(box, h, w):
    y, x, dy, dx = box
    need(
        integer(y, 0, h - 1)
        and integer(x, 0, w - 1)
        and integer(dy, 1, h - y)
        and integer(dx, 1, w)
        and dy * dx <= MAX_CELLS,
        "invalid ROI; rows do not wrap, columns may wrap once",
    )
    return y, x, dy, dx


def member(i, name):
    return f"band-{i:04}-{name}"


@contextmanager
def destination(path, sources):
    if path is None:
        yield None
        return
    path = Path(path).resolve()
    need(os.name == "nt", "receipt publisher requires Windows no-replace directory rename")
    need(
        not path.exists()
        and all(not path.is_relative_to(s) and not s.is_relative_to(path) for s in sources),
        "receipt destination must be fresh and separate",
    )
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".roi-") as tmp:
        stage = Path(tmp) / "receipt"
        stage.mkdir()
        yield stage
        os.rename(stage, path)


def inspect(bundle, idx, parcel_pin, index_pin, box, *, dest=None, detached=False):
    """Read only intersecting blocks, return counts and ordered crop hashes. Never deduce global integrity."""
    bundle, idx = Path(bundle).resolve(), Path(idx).resolve()
    doc, info, manifest, raw = metadata(bundle, idx, parcel_pin, index_pin)
    h, w = doc["shape_ha"]
    y, x, dy, dx = area(box, h, w)
    band = info["block_rows"]
    first, last = y // band, (y + dy - 1) // band
    blocks = range(first, last + 1)
    names = {"manifest.json", "index.json", "receipt.json"} | {
        member(i, n) for i in blocks for n, _, _ in PAYLOADS
    }
    if detached:
        need(
            dest is None
            and idx == bundle / "index.json"
            and {p.name for p in bundle.iterdir()} == names,
            "extra or missing receipt members",
        )
    sums = [hashlib.sha256(), hashlib.sha256()]
    support = [0, 0, 0]
    ids = Counter()
    split = min(w - x, dx)
    covered = 0
    with destination(dest, (bundle, idx)) as stage, ExitStack() as stack:
        if stage is not None:
            (stage / "manifest.json").write_bytes(manifest)
            (stage / "index.json").write_bytes(raw)
        fs = (
            []
            if detached
            else [stack.enter_context(regular(bundle / n).open("rb")) for n, _, _ in PAYLOADS]
        )
        for i in blocks:
            start = i * band
            end = min(start + band, h)
            covered += (end - start) * w
            data = []
            for col, (name, size, _) in enumerate(PAYLOADS):
                count = (end - start) * w * size
                fn = member(i, name)
                if detached:
                    buf = bounded(bundle / fn, count)
                else:
                    fs[col].seek(start * w * size)
                    buf = fs[col].read(count)
                need(
                    len(buf) == count and digest(buf) == info["blocks"][i][col],
                    "invalid touched block: " + fn,
                )
                if stage is not None:
                    (stage / fn).write_bytes(buf)
                data.append(memoryview(buf))
            for yy in range(max(y, start), min(y + dy, end)):
                off = (yy - start) * w
                for a, b in ((off + x, off + x + split), (off, off + dx - split)):
                    if b == a:
                        continue
                    m = data[0][a:b]
                    labels = data[1][4 * a : 4 * b]
                    sums[0].update(m)
                    sums[1].update(labels)
                    for j, (k, (v,)) in enumerate(
                        zip(m, struct.iter_unpack("<i", labels), strict=True)
                    ):
                        need(
                            (k == 0 and v == 0)
                            or (k == 2 and v == -1)
                            or (k == 1 and 1 <= v <= start * w + a + j + 1),
                            "invalid mask/ID support",
                        )
                        support[k] += 1
                        if k == 1:
                            ids[v] += 1
        out = dict(
            format=RECEIPT,
            scope="selected rectangle only; complete original parcel not checked by reader",
            parcel_manifest_sha256=parcel_pin,
            index_sha256=index_pin,
            edge_manifest_sha256=doc["edge_manifest_sha256"],
            pair=doc["pair"],
            shape_ha=[h, w],
            box_y_x_h_w=[y, x, dy, dx],
            wraps_azimuth=dx > split,
            authenticated_rows_half_open=[first * band, min((last + 1) * band, h)],
            authenticated_payload_bytes=covered * 5,
            requested_cells=dy * dx,
            below_cells=support[0],
            observed_cells=support[1],
            unknown_cells=support[2],
            observed_cells_by_global_id={str(k): v for k, v in sorted(ids.items())},
            mask_u8_sha256=sums[0].hexdigest(),
            labels_i32le_sha256=sums[1].hexdigest(),
        )
        payload = encode(out)
        if stage is not None:
            (stage / "receipt.json").write_bytes(payload)
        if detached:
            need(
                bounded(bundle / "receipt.json", len(payload)) == payload,
                "receipt does not describe requested ROI",
            )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for mode in ("index", "query", "pack", "verify"):
        p = sub.add_parser(mode)
        p.add_argument("bundle", type=Path)
        if mode != "verify":
            p.add_argument("index", type=Path)
        if mode == "pack":
            p.add_argument("out", type=Path)
        p.add_argument("--parcel-pin", required=True)
        if mode == "index":
            p.add_argument("--block-rows", type=int, default=4)
        else:
            p.add_argument("--index-pin", required=True)
            p.add_argument(
                "--box", type=int, nargs=4, required=True, metavar=("Y", "X", "H", "W")
            )
    a = parser.parse_args()
    if a.action == "index":
        print(index(a.bundle, a.index, a.parcel_pin, a.block_rows))
    else:
        detached = a.action == "verify"
        result = inspect(
            a.bundle,
            a.bundle / "index.json" if detached else a.index,
            a.parcel_pin,
            a.index_pin,
            a.box,
            dest=getattr(a, "out", None),
            detached=detached,
        )
        print(json.dumps(result, indent=2))
