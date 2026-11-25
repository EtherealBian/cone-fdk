"""Read-only, standard-library anchor-to-anchor audit on a pinned CFDK offline console."""

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from array import array
from collections import Counter, deque
from itertools import product
from pathlib import Path

MAX_FILE = 64 << 20
MAX_CELLS = 1 << 20
MAX_PAIR_CELLS = 8 << 20
MAX_TURNS = 256
TAG = b'<script id="payload" type="application/json">'
PIN = re.compile(r"[0-9a-f]{64}")


def need(test, why):
    if not test:
        raise ValueError(why)


def pin(s):
    need(isinstance(s, str) and PIN.fullmatch(s), "expected an external lowercase SHA-256")
    return s


def load_page(path, expected):
    pin(expected)
    with path.open("rb") as f:
        raw = f.read(MAX_FILE + 1)
    need(
        len(raw) <= MAX_FILE and hashlib.sha256(raw).hexdigest() == expected,
        "console size or SHA-256 mismatch",
    )
    need(raw.count(TAG) == 1, "missing or ambiguous console JSON block")
    _, _, body = raw.partition(TAG)
    block, close, _ = body.partition(b"</script>")
    need(bool(close), "unterminated console payload")
    doc = json.loads(block)
    need(doc["format"] == "CFDK-region-console-v1", "unknown console format")
    return doc


def decode(s, n):
    """Expand only within the declared small grid, then verify original little-endian bytes."""
    codec, text = s["pixels"]
    need(codec in ("i32le", "rle32le") and isinstance(text, str), "unknown pixel encoding")
    cap = 4 * n if codec == "i32le" else 8 * n
    need(len(text) <= 4 * ((cap + 2) // 3), "encoded data exceeds grid")
    src = base64.b64decode(text, validate=True)
    if codec == "i32le":
        need(len(src) == 4 * n, "raw length differs")
        data = src
    else:
        import struct

        need(0 < len(src) <= cap and len(src) % 8 == 0, "invalid RLE length")
        data = bytearray()
        for length, value in struct.iter_unpack("<ii", src):
            need(0 < length <= n - len(data) // 4, "invalid RLE run")
            data.extend(struct.pack("<i", value) * length)
        need(len(data) == 4 * n, "truncated RLE")
    need(
        hashlib.sha256(data).hexdigest() == pin(s["labels_i32le_sha256"]),
        "label map SHA-256 mismatch",
    )
    labels = array("i")
    need(labels.itemsize == 4, "32-bit int required")
    labels.frombytes(data)
    if sys.byteorder != "little":
        labels.byteswap()
    need(all(-1 <= v <= n for v in labels), "invalid label ID")
    need(
        labels.count(-1) == s["unknown_cells"]
        and n - labels.count(-1) - labels.count(0) == s["observed_flagged_cells"],
        "coverage disagrees with decoded map",
    )
    return labels


def adjacent(p, w, n):
    x = p % w
    yield p + 1 if x + 1 < w else p + 1 - w, "E"
    yield p - 1 if x else p + w - 1, "W"
    if p >= w:
        yield p - w, "N"
    if p + w < n:
        yield p + w, "S"


def path_digest(path):
    buf = array("i", path)
    if sys.byteorder != "little":
        buf.byteswap()
    return hashlib.sha256(buf.tobytes()).hexdigest()


def route(labels, w, a, b):
    """Exact minimum unknown *cells* on a candidate path, including unknown endpoints.

    Below-gate cells are impassable. Positive labels cost zero, unknown cells cost one.
    Only the angular coordinate wraps. A zero-cost path is observed; a positive-cost path
    is a hypothetical completion, never promoted to measured evidence.
    """
    n = len(labels)
    need(
        3 <= w <= n and n % w == 0 and 0 <= a < n and 0 <= b < n and a != b,
        "invalid grid or anchors",
    )
    va, vb = labels[a], labels[b]
    res = dict(anchor_label_ids=[va, vb], state=None, min_unknown_cells=None, path=None)
    if va == 0 or vb == 0:
        res["state"] = "anchor_below_gate"
        return res
    inf = n + 1
    dist = array("i", [inf]) * n
    done = bytearray(n)
    dist[a] = int(va < 0)
    q = deque([a])
    while q:
        u = q.popleft()
        if done[u]:
            continue
        if dist[u] > dist[b]:
            break
        done[u] = 1
        for v, _ in adjacent(u, w, n):
            if done[v] or labels[v] == 0:
                continue
            cost = int(labels[v] < 0)
            d = dist[u] + cost
            if d < dist[v]:
                dist[v] = d
                (q.append if cost else q.appendleft)(v)
    need(
        not (va > 0 and vb > 0) or ((va == vb) == (dist[b] == 0)),
        "anchor labels contradict measured connectivity",
    )
    if dist[b] == inf:
        res["state"] = "separated_on_grid"
        return res
    res["min_unknown_cells"] = dist[b]
    res["state"] = "observed_link" if dist[b] == 0 else "unresolved_blind_path"
    # Among minimum-unknown paths pick the fewest steps. An ordinary BFS on edges
    # that preserve the optimal cost does not conflate a long safe route with a short blind one.
    parent = array("i", [-1]) * n
    parent[a] = a
    q = deque([a])
    while q and parent[b] == -1:
        u = q.popleft()
        for v, _ in adjacent(u, w, n):
            if parent[v] == -1 and done[v] and dist[v] == dist[u] + int(labels[v] < 0):
                parent[v] = u
                q.append(v)
    need(parent[b] != -1, "missing minimum-cost path")
    path = array("i", [b])
    v = b
    while v != a:
        v = parent[v]
        path.append(v)
    path.reverse()
    need(sum(labels[x] < 0 for x in path) == dist[b], "invalid path witness")
    turns = []
    last = None
    count = 0
    for i in range(1, len(path)):
        heading = next(k for v, k in adjacent(path[i - 1], w, n) if v == path[i])
        if heading == last:
            if count <= MAX_TURNS:
                turns[-1][1] += 1
        else:
            last = heading
            count += 1
            if count <= MAX_TURNS:
                turns.append([heading, 1])
    res["path"] = dict(
        cells=len(path),
        row_major_i32le_sha256=path_digest(path),
        turn_runs=count,
        direction_runs=turns if count <= MAX_TURNS else None,
    )
    return res


def audit(page, sha256, entry_id, anchors):
    doc = load_page(page, sha256)
    entries = [e for e in doc["entries"] if e["id"] == entry_id]
    need(len(entries) == 1, "unknown or duplicate entry")
    e = entries[0]
    need(
        e["state"] in ("complete_coverage", "incomplete_coverage"),
        "no measured maps for this entry",
    )
    h, w = e["shape"]
    need(
        all(type(v) is int and v > 0 for v in (h, w)) and w >= 3 and h * w <= MAX_CELLS,
        "grid size limit",
    )
    levels = e["levels"]
    k = len(levels)
    need(1 <= k <= 32 and h * w * k * k <= MAX_PAIR_CELLS, "pair size limit")
    need(
        all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in levels)
        and sorted(set(levels)) == levels,
        "invalid global levels",
    )
    nominal = e["nominal"]
    need(
        len(nominal) == 2
        and all(type(v) is int and 0 <= v < k for v in nominal)
        and nominal[0] == nominal[1],
        "invalid nominal pair",
    )
    gate = e["gate"]
    need(type(gate) in (int, float) and math.isfinite(gate) and gate > 0, "invalid loss gate")
    sc = e["scenarios"]
    need(
        len(sc) == k * k
        and all(len(r["pair"]) == 2 and all(type(v) is int for v in r["pair"]) for r in sc)
        and sorted(tuple(r["pair"]) for r in sc) == list(product(range(k), repeat=2)),
        "not exactly one of every global pair",
    )
    need(
        len(anchors) == 2
        and len({tuple(t) for t in anchors}) == 2
        and all(
            len(t) == 2 and all(type(v) is int for v in t) and 0 <= t[0] < h and 0 <= t[1] < w
            for t in anchors
        ),
        "two distinct zero-based anchors inside grid required",
    )
    a, b = (y * w + x for y, x in anchors)
    rows = []
    for s in sorted(sc, key=lambda r: r["pair"]):
        labels = decode(s, h * w)
        rows.append(
            dict(
                pair=s["pair"],
                labels_i32le_sha256=s["labels_i32le_sha256"],
                **route(labels, w, a, b),
            )
        )
    counts = dict(sorted(Counter(s["state"] for s in rows).items()))
    return dict(
        format="CFDK-anchor-audit-v1",
        console_sha256=sha256,
        edge_sha256=pin(e["source_sha256"]),
        region_sha256=pin(e["result_sha256"]),
        entry=entry_id,
        shape_ha=[h, w],
        anchors_row_col=anchors,
        levels_per_mm=levels,
        loss_gate_mm=e["gate"],
        nominal_pair=e["nominal"],
        counts=counts,
        scenarios=rows,
    )


def publish(out, doc):
    data = (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("utf8")
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=out.parent, delete=False) as f:
            tmp = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(
            tmp, out
        )  # Atomic no-overwrite publication on NTFS and POSIX filesystems supporting hard links.
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("page", type=Path)
    p.add_argument("entry")
    p.add_argument("out", type=Path)
    p.add_argument("--console-sha256", required=True)
    for key in ("start", "end"):
        p.add_argument("--" + key, type=int, nargs=2, metavar=("ROW", "COL"), required=True)
    a = p.parse_args()
    try:
        d = audit(a.page, a.console_sha256, a.entry, [a.start, a.end])
        publish(a.out, d)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as err:
        p.exit(1, f"anchor audit failed: {err}\n")
    print(json.dumps(d["counts"]))
