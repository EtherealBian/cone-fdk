"""Portable, script-free spatial review sheet of one verified threshold atlas case. Standard library only."""

import argparse
import base64
import hashlib
import os
import stat
import struct
import sys
import types
import zlib
from html import escape
from pathlib import Path

ATLAS = "e7037bd30642430d3f3cb50d7c4617bc692ecfe98212430712eda3ac414ba70e"


def loader():
    path = Path(__file__).with_name("threshold_atlas.py")
    with path.open("rb") as f:
        b = f.read((128 << 10) + 1)
    if hashlib.sha256(b).hexdigest() != ATLAS:
        raise ValueError("unrecognized trusted atlas")
    mod = types.ModuleType("pinned_threshold_atlas")
    mod.__file__ = str(path)
    exec(compile(b, str(path), "exec"), mod.__dict__)
    return mod


a = loader()
c = a.c
need = c.need
# Indexed PNG stores actual counts, independent of display palette: 10*flagged + unknown.
COLORS = ("#dfe8e6", "#d9a125", "#b53d36", "#92a3bb", "#6c508c", "#343d53")
DESCRIPTIONS = (
    "Always below",
    "Mixed, fully observed",
    "Always flagged",
    "Partial, no flags",
    "Partial, with flags",
    "Always unknown",
)


def category(f, u, n):
    need(
        all(type(x) is int for x in (f, u, n))
        and 1 <= n <= 9
        and 0 <= f
        and 0 <= u
        and f + u <= n,
        "invalid vote",
    )
    return 5 if u == n else (4 if f else 3) if u else 0 if not f else 2 if f == n else 1


def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))


def png(raw, w, h, n):
    need(
        1 <= w <= 1440 and 1 <= h <= 512 and len(raw) == w * h and len(raw) <= a.c.MAX_CELLS,
        "raster bounds",
    )
    palette = bytearray(bytes.fromhex("ff00ff") * 100)
    for f in range(n + 1):
        for u in range(n - f + 1):
            k = f * 10 + u
            palette[k * 3 : k * 3 + 3] = bytes.fromhex(COLORS[category(f, u, n)][1:])
    need(all(v // 10 + v % 10 <= n for v in raw), "impossible vote index")
    scan = b"".join(b"\0" + raw[y * w : (y + 1) * w] for y in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 3, 0, 0, 0))
        + chunk(b"PLTE", palette)
        + chunk(b"IDAT", zlib.compress(scan, 9))
        + chunk(b"IEND", b"")
    )


def indices(h, w, roi):
    if roi is None:
        rh = min(15, h)
        rw = min(24, w)
        roi = ((h - rh) // 2, (w - rw // 2) % w, rh, rw)
    y, x, rh, rw = roi
    need(
        all(type(t) is int for t in roi)
        and 0 <= x < w
        and 0 <= y < h
        and 1 <= rh <= h - y
        and 1 <= rw <= w,
        "ROI out of range (rows never wrap; columns at most once)",
    )
    return y, x, rh, rw


def crop(raw, h, w, roi):
    y, x, rh, rw = indices(h, w, roi)
    need(len(raw) == h * w, "source size")
    return bytes(raw[(y + j) * w + (x + i) % w] for j in range(rh) for i in range(rw))


def spatial(f, root, name, roi=None):
    # The atlas rechecks every parcel, ledger, source gate and EOF. Read the masks a second time against the SAME external root and cross-check the ordered vote digest.
    atlas = a.atlas(f, root)
    need(name in atlas["cases"], "unknown or upstream-refused case")
    case = atlas["cases"][name]
    h, w = case["shape_ha"]
    need(h <= 512 and w <= 1440, "display budget")
    roi = indices(h, w, roi)
    f.seek(0)
    r = c.Reader(f)
    chain = c.obj(r.get(c.pin(root), c.MAX_DOC))
    r.get(chain["spark_survey_sha256"], c.MAX_DOC)
    yes = bytearray(h * w)
    blind = bytearray(h * w)
    keys = []
    for row in chain["rows"]:
        ident, i, j = c.ident(row)
        parts = [r.get(row[k], limit) for k, limit in zip(c.SLOTS, a.LIMITS, strict=True)]
        if ident == name:
            keys.append([i, j])
            m = parts[3]
            need(len(m) == len(yes) and all(t <= 2 for t in m), "mask size or classes")
            for ix, t in enumerate(m):
                yes[ix] += t == 1
                blind[ix] += t == 2
    n = len(keys)
    need(
        not f.read(1) and sorted(keys) == case["ordered_pairs"] and 1 <= n <= 9,
        "different setting inventory",
    )
    votes = bytearray(2 * h * w)
    votes[::2] = yes
    votes[1::2] = blind
    need(c.sha(votes) == case["votes_sha256"], "spatial vote provenance")
    raw = bytes(10 * v + b for v, b in zip(yes, blind, strict=True))
    zoom = crop(raw, h, w, roi)
    y, x, rh, rw = roi
    src = case["source"]
    d = dict(
        format="CFDK-spatial-vote-sheet-v1",
        case=name,
        source_chain_sha256=root,
        atlas_sha256=c.sha(c.encoded(atlas) + b"\n"),
        atlas_code_sha256=ATLAS,
        atlas_protocol_sha256=a.PROTOCOL,
        votes_sha256=case["votes_sha256"],
        shape_row_col=[h, w],
        settings=n,
        pairs=case["ordered_pairs"],
        anchors_row_col=src["anchors_row_col"],
        observed_link_settings=case["observed_link_settings"],
        partition=case["partition"],
        vote_histogram=case["vote_counts"],
        roi_row_col_height_width=list(roi),
        roi_columns_wrap=x + rw > w,
        roi_values_sha256=c.sha(zoom),
        cell_encoding="indexed PNG palette index = 10 * flagged setting count + unknown setting count; row 0 at top; only columns wrap",
        scope="Simulated sampled grid. Votes count settings, not defect probability or physical dimensions. A pooled path need not exist at one setting.",
    )
    return d, png(raw, w, h, n), png(zoom, rw, rh, n)


def sheet(doc, full, detail):
    h, w = doc["shape_row_col"]
    y, x, rh, rw = doc["roi_row_col_height_width"]
    name = doc["case"]
    n = doc["settings"]
    links = len(doc["observed_link_settings"])
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="1200" height="720" viewBox="0 0 1200 720" role="img" aria-labelledby="title desc">',
        f'<title id="title">Threshold vote sheet: {escape(name)}</title>',
        '<desc id="desc">Lossless cell-index overview and unrolled detail. Only columns wrap. Unknown is distinct from below threshold. Numerical setting counts do not establish physical accuracy.</desc>',
        f'<metadata id="evidence">{escape(c.encoded(doc).decode())}</metadata>',
        '<rect width="1200" height="720" fill="#faf9f5"/>',
        '<g font-family="DejaVu Sans, sans-serif" fill="#172f35">',
    ]

    def text(x, y, s, size=15, color="#172f35", weight="normal"):
        parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(str(s))}</text>'
        )

    def panel(id, b, x, y, w, h, max_w, max_h):
        scale = min(max_w / w, max_h / h)
        scale = int(scale) if scale >= 1 else scale
        sw = w * scale
        sh = h * scale
        parts.append(
            f'<image id="{id}" x="{x}" y="{y}" width="{sw:.6f}" height="{sh:.6f}" preserveAspectRatio="none" style="image-rendering:pixelated" xlink:href="data:image/png;base64,{base64.b64encode(b).decode()}"/>'
        )
        return scale

    text(42, 39, "CONE FDK  /  OFFLINE FIELD NOTES", 14, "#396b65", "bold")
    text(42, 90, name, 42, weight="bold")
    text(860, 49, "GLOBAL THRESHOLD SETTINGS", 13)
    text(860, 86, f"{n} settings  /  {links} anchor links", 22, weight="bold")
    text(
        42,
        128,
        f"01   COMPLETE GRID     /     {h} axial rows by {w} azimuth columns",
        16,
        weight="bold",
    )
    panel("full", full, 42, 148, w, h, 1090, 139)
    text(
        42,
        312,
        f"Column 0 at left; column {w - 1} at right. The two vertical edges are neighbors. Rows do not wrap.",
        15,
    )
    text(42, 362, "02   CELL DETAIL", 17, weight="bold")
    text(
        42,
        389,
        f"Row {y} to {y + rh - 1}  /  col {x} to {(x + rw - 1) % w}"
        + ("  (wraps)" if doc["roi_columns_wrap"] else ""),
        15,
    )
    s = panel("detail", detail, 42, 411, rw, rh, 415, 180)
    if doc["roi_columns_wrap"]:
        seam = 42 + (w - x) * s
        parts.append(
            f'<path id="azimuth-seam" d="M {seam:.6f} 408 v -6" stroke="#172f35" stroke-width="2"/>'
        )
    text(530, 362, "03   HOW TO READ THE MAP", 17, weight="bold")
    for i, label in enumerate(DESCRIPTIONS):
        px = 530 + (i // 3) * 299
        py = 402 + (i % 3) * 35
        parts.append(f'<rect x="{px}" y="{py - 13}" width="17" height="17" fill="{COLORS[i]}"/>')
        text(px + 27, py, label, 16)
    anchors = doc["anchors_row_col"]
    text(
        530,
        535,
        "Anchor A (row, col): " + str(tuple(anchors[0])) + "     B: " + str(tuple(anchors[1])),
        15,
    )
    text(530, 562, "Links count settings with an actual A-to-B path.", 15)
    text(530, 589, "Missing observations are never counted as below threshold.", 15)
    parts.append('<path d="M42 620H1158" fill="none" stroke="#b6c5c1"/>')
    text(
        42,
        652,
        "SIMULATED GRID   /   Counts describe threshold settings, not probability, physical size or defect accuracy.",
        16,
        weight="bold",
    )
    text(42, 684, "Source chain SHA-256: " + doc["source_chain_sha256"], 13)
    parts.append("</g></svg>")
    return ("\n".join(parts) + "\n").encode()


def render(path, root, name, roi=None):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        need(
            stat.S_ISREG(st.st_mode) and 0 < st.st_size <= c.MAX_PACK,
            "bounded regular bundle required",
        )
        return sheet(*spatial(f, root, name, roi))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pack", type=Path)
    cli.add_argument("external_chain_sha256")
    cli.add_argument("case")
    cli.add_argument("--roi", type=int, nargs=4, metavar=("ROW", "COL", "HEIGHT", "WIDTH"))
    args = cli.parse_args()
    try:
        sys.stdout.buffer.write(
            render(args.pack, args.external_chain_sha256, args.case, args.roi)
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        OverflowError,
        RecursionError,
    ) as e:
        cli.exit(1, f"spatial sheet refused: {e}\n")


if __name__ == "__main__":
    main()
