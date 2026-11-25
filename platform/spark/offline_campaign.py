"""Detached review of a pinned periodic campaign. Standard library only; never extracts files.

Wire order: magic, chain, survey, then parcel/face/index/mask/labels/face bytes for
all jobs in the pinned chain's order. A frame is u32le length + raw bytes, and is
stored only on first occurrence of its expected SHA-256. There are no paths, codecs,
executable members, self-issued trust roots, or external references to chase.
"""

import hashlib
import json
import os
import re
import stat
import struct
import sys
from argparse import ArgumentParser
from array import array
from collections import deque
from functools import partial
from pathlib import Path

MAGIC = b"CFDK-OFFLINE-CAMPAIGN\x00\x01"
ROI = "a05630e04c348abf03897082c74740d7ac00a4ad3246ab7094873089f58530a0"
FACE = "72ba73a2a80783466ac3fad9c0770455ef3e2d373471de5ddcf17b70c89cb2f1"
MAX_PACK, MAX_DOC, MAX_CELLS, MAX_TOTAL = 256 << 20, 4 << 20, 1 << 18, 32 << 20
SLOTS = (
    "new_manifest_sha256",
    "face_manifest_sha256",
    "index_sha256",
    "mask_sha256",
    "labels_sha256",
    "face_sha256",
)


def need(ok, why):
    if not ok:
        raise ValueError(why)


def integer(v, low, high):
    return type(v) is int and low <= v <= high


def pin(x):
    need(type(x) is str and re.fullmatch("[a-f0-9]{64}", x), "expected external SHA-256")
    return x


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "ascii"
    )


def obj(data):
    def pairs(items):
        out = {}
        for k, v in items:
            need(k not in out, "duplicate JSON key")
            out[k] = v
        return out

    def invalid(_):
        raise ValueError("non-finite JSON")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def ident(row):
    name, pair = row["case"], row["pair"]
    need(type(name) is str and re.fullmatch("[a-z0-9][a-z0-9-]{0,63}", name), "case name")
    need(
        type(pair) is list and len(pair) == 2 and all(integer(i, 0, 2) for i in pair),
        "threshold pair",
    )
    return name, *pair


def plan(chain, survey):
    need(
        chain["format"] == "CFDK-GB10-source-to-interface-v1"
        and survey["format"] == "CFDK-periodic-parcel-campaign-v1",
        "campaign format",
    )
    for key in (
        "label_qualification_sha256",
        "interface_qualification_sha256",
        "source_recipe_sha256",
        "spark_survey_sha256",
    ):
        pin(chain[key])
    need(
        survey["recipe_sha256"] == chain["source_recipe_sha256"]
        and survey["qualification_sha256"] == chain["label_qualification_sha256"],
        "survey provenance",
    )
    rows = chain["rows"]
    cases = survey["cases"]
    withheld = {}
    seen = set()
    grid = {}
    need(
        type(rows) is list
        and integer(len(rows), 1, 512)
        and chain["maps"] == survey["maps"] == len(rows),
        "map count",
    )
    need(
        integer(chain["cells"], 1, MAX_TOTAL) and chain["cells"] == survey["pair_cells"],
        "campaign size",
    )
    need(type(cases) is dict and 1 <= len(cases) <= 128, "case count")
    for case, record in cases.items():
        ident({"case": case, "pair": [0, 0]})
        status = record["status"]
        if status == "upstream_refused":
            need(
                set(record) == {"status", "reason"}
                and type(record["reason"]) is str
                and record["reason"],
                "refusal record",
            )
            withheld[case] = record["reason"]
            continue
        need(
            status in ("complete_coverage", "incomplete_coverage")
            and 1 <= len(record["rows"]) <= 9,
            "case coverage",
        )
        pairs = set()
        for s in record["rows"]:
            key = ident(dict(case=case, pair=s["pair"]))
            need(key not in grid, "duplicate survey job")
            grid[key] = s
            pairs.add(key[1:])
        # Coverage is an upstream material/visibility property, not a missing-pair count.
        span = max(max(x) for x in pairs) + 1
        need(
            pairs == {(a, b) for a in range(span) for b in range(span)},
            "incomplete threshold product",
        )
    need(
        type(chain["upstream_refusals"]) is list
        and sorted(chain["upstream_refusals"]) == sorted(withheld),
        "refusal inventory",
    )
    for row in rows:
        key = ident(row)
        need(key in grid and key not in seen, "unknown, refused or duplicate job")
        seen.add(key)
        s = grid[key]
        for slot in (*SLOTS, "old_manifest_sha256"):
            pin(row[slot])
        need(
            row["old_manifest_sha256"] != row["new_manifest_sha256"]
            and (row[SLOTS[0]], row[SLOTS[3]], row[SLOTS[4]])
            == (s["manifest_sha256"], s["mask_sha256"], s["labels_sha256"]),
            "wrong lineage",
        )
    need(seen == set(grid), "missing job")
    return grid, withheld


class Reader:
    def __init__(self, stream):
        self.f = stream
        self.offsets = {}
        self.logical = 0
        need(stream.read(len(MAGIC)) == MAGIC, "invalid file signature")

    def get(self, expected, limit):
        pin(expected)
        f = self.f
        pos = f.tell()
        first = expected not in self.offsets
        if first:
            head = f.read(4)
            need(len(head) == 4, "missing frame")
            (size,) = struct.unpack("<I", head)
            self.offsets[expected] = (f.tell(), size)
        else:
            offset, size = self.offsets[expected]
            f.seek(offset)
        need(
            0 < size <= limit
            and f.tell() + size <= MAX_PACK
            and self.logical + size <= 2 * MAX_PACK,
            "frame size or logical campaign budget",
        )
        data = f.read(size)
        need(len(data) == size and sha(data) == expected, "frame digest or length")
        self.logical += size
        if not first:
            f.seek(pos)
        return data


def bind(chain, row, parts):
    p, f, x = map(obj, parts[:3])
    mask, labels, faces = parts[3:]
    pair = row["pair"]
    h, w = p["shape_ha"]
    need(integer(h, 1, 4096) and integer(w, 3, 4096) and h * w <= MAX_CELLS, "bounded grid shape")
    n = h * w
    need((len(mask), len(labels), len(faces)) == (n, 4 * n, n), "payload shape")
    need(
        p["format"] == "CFDK-periodic-label-parcel-v1"
        and p["pair"] == pair
        and p["qualification_sha256"] == chain["label_qualification_sha256"],
        "parcel parent",
    )
    need(
        f["format"] == "CFDK-certified-interface-ledger-v1"
        and f["protocol_sha256"] == FACE
        and f["qualification_sha256"] == chain["interface_qualification_sha256"],
        "face parent",
    )
    m = f["measurement"]
    c = p["certificate"]
    sh = [h, w]
    parent = row[SLOTS[0]]
    need(
        m["parcel_manifest_sha256"] == x["parcel_manifest_sha256"] == parent
        and m["edge_manifest_sha256"] == pin(p["edge_manifest_sha256"])
        and m["pair"] == pair
        and m["shape_ha"] == x["shape_ha"] == sh,
        "wrong measurement parent",
    )
    need(
        (c["mask_u8_sha256"], c["labels_i32le_sha256"], m["summary"]["face_u8_sha256"])
        == tuple(row[k] for k in SLOTS[3:]),
        "wrong dense payload",
    )
    need(
        x["format"] == "CFDK-periodic-row-index-v1" and x["protocol_sha256"] == ROI,
        "row index provenance",
    )
    band = x["block_rows"]
    need(integer(band, 1, h), "row index band")
    blocks = [
        [sha(mask[y * w : (y + band) * w]), sha(labels[y * w * 4 : (y + band) * w * 4])]
        for y in range(0, h, band)
    ]
    need(x["blocks"] == blocks, "row index contents")
    return p, m["summary"]


def neighbours(i, w, n):
    y, x = divmod(i, w)
    start = y * w
    return (
        i - w if i >= w else -1,
        start + (x + 1) % w,
        i + w if i + w < n else -1,
        start + (x - 1) % w,
    )


def audit(mask, raw_labels, faces, p, summary, survey_row):
    h, w = p["shape_ha"]
    n = h * w
    seen = bytearray(n)
    labels = array("i")
    labels.frombytes(raw_labels)
    need(labels.itemsize == 4, "int32 platform required")
    if sys.byteorder != "little":
        labels.byteswap()
    comps = []
    totals = [[0] * 3 for _ in range(4)]
    queue = deque()
    flagged = 0
    unknown = 0
    for start, code in enumerate(mask):
        need(code in (0, 1, 2), "non-ternary mask")
        if code != 1:
            need(
                labels[start] == (0 if code == 0 else -1) and faces[start] == 0,
                "non-defect support",
            )
            unknown += code == 2
            continue
        if seen[start]:
            continue
        need(len(comps) < 32768, "component table budget")
        area = inside = 0
        tab = [[0] * 3 for _ in range(4)]
        queue.append(start)
        seen[start] = 1
        while queue:
            i = queue.popleft()
            area += 1
            need(labels[i] == start + 1, "not canonical periodic component ID")
            byte = 0
            for d, j in enumerate(neighbours(i, w, n)):
                if j >= 0 and mask[j] == 1:
                    inside += d in (1, 2)
                    if not seen[j]:
                        seen[j] = 1
                        queue.append(j)
                else:
                    t = 3 if j < 0 else (1 if mask[j] == 0 else 2)
                    tab[d][t - 1] += 1
                    totals[d][t - 1] += 1
                    byte |= t << (2 * d)
            need(byte == faces[i], "face byte disagrees with domain")
        need(4 * area == 2 * inside + sum(map(sum, tab)), "cell-face conservation")
        flagged += area
        comps.append(
            dict(id=start + 1, area_cells=area, internal_undirected_edges=inside, faces=tab)
        )
    rebuilt = dict(
        directions=["north", "east", "south", "west"],
        face_kinds=["measured_below", "unknown", "axial_cut"],
        components=comps,
        faces_by_direction_and_kind=totals,
        observed_cells=flagged,
        source_cells=n,
        face_u8_sha256=sha(faces),
    )
    need(encoded(rebuilt) == encoded(summary), "face ledger counts or IDs")
    cert = p["certificate"]
    need(
        encoded(
            [
                cert[k]
                for k in (
                    "observed_components",
                    "observed_flagged_cells",
                    "unknown_cells",
                    "verified_cells",
                )
            ]
        )
        == encoded([len(comps), flagged, unknown, n]),
        "certificate census",
    )
    anchors = p["anchors_row_col"]
    need(
        len(anchors) == 2
        and anchors[0] != anchors[1]
        and all(
            len(t) == 2 and integer(t[0], 0, h - 1) and integer(t[1], 0, w - 1) for t in anchors
        ),
        "anchor positions",
    )
    a, b = (y * w + x for y, x in anchors)
    endpoints = [("below", "flagged", "unknown")[mask[i]] for i in (a, b)]
    if "below" in endpoints:
        state = "anchor_below_gate"
    elif labels[a] > 0 and labels[a] == labels[b]:
        state = "observed_link"
    else:
        possible = bytearray(n)
        possible[a] = 1
        queue.append(a)
        while queue and not possible[b]:
            for j in neighbours(queue.popleft(), w, n):
                if j >= 0 and mask[j] != 0 and not possible[j]:
                    possible[j] = 1
                    queue.append(j)
        state = "unresolved_blind_path" if possible[b] else "separated_on_grid"
    scenario = dict(
        pair=p["pair"],
        observed_components=len(comps),
        largest_observed_cells=max((c["area_cells"] for c in comps), default=0),
        observed_flagged_cells=flagged,
        unknown_cells=unknown,
        anchor_cells=endpoints,
        anchor_state=state,
    )
    need(
        encoded(p["source_scenario"]) == encoded(scenario)
        and survey_row["anchor_state"] == state,
        "anchor/scenario mismatch",
    )
    return scenario


def summary_row(row, p, summary):
    return dict(
        case=row["case"],
        pair=row["pair"],
        parcel_sha256=row[SLOTS[0]],
        face_manifest_sha256=row[SLOTS[1]],
        index_sha256=row[SLOTS[2]],
        source_cells=summary["source_cells"],
        face_table_sha256=sha(encoded(summary)),
        scenario=p["source_scenario"],
    )


def walk(stream, root_pin, *, checkpoint=None):
    r = Reader(stream)
    chain = obj(r.get(pin(root_pin), MAX_DOC))
    survey = obj(r.get(chain["spark_survey_sha256"], MAX_DOC))
    grid, withheld = plan(chain, survey)
    rows = []
    total = 0
    for row in chain["rows"]:
        parts = [
            r.get(row[slot], limit)
            for slot, limit in zip(
                SLOTS,
                (MAX_DOC, MAX_DOC, MAX_DOC, MAX_CELLS, 4 * MAX_CELLS, MAX_CELLS),
                strict=True,
            )
        ]
        p, table = bind(chain, row, parts)
        result = summary_row(row, p, table)
        total += len(parts[3])
        need(total <= MAX_TOTAL, "campaign cell budget")

        compute = partial(audit, *parts[3:], p, table, grid[ident(row)])

        if checkpoint is None:
            compute()
        else:
            checkpoint(row, result, compute)
        rows.append(result)
    need(total == chain["cells"] and not stream.read(1), "total size or trailing frames")
    return dict(
        format="CFDK-detached-periodic-review-v1",
        chain_sha256=root_pin,
        source_recipe_sha256=chain["source_recipe_sha256"],
        scope=chain["scope"],
        maps=len(rows),
        cells=total,
        unique_frames=len(r.offsets),
        logical_payload_bytes=r.logical,
        refused=withheld,
        rows=rows,
    )


def review(path, root_pin, *, checkpoint=None):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        need(
            stat.S_ISREG(st.st_mode) and 0 < st.st_size <= MAX_PACK,
            "expected bounded regular pack",
        )
        return walk(f, root_pin, checkpoint=checkpoint)


if __name__ == "__main__":
    a = ArgumentParser(description=__doc__)
    a.add_argument("pack", type=Path)
    a.add_argument("chain_sha256")
    args = a.parse_args()
    try:
        print(encoded(review(args.pack, args.chain_sha256)).decode("ascii"))
    except (ValueError, OSError, KeyError, IndexError, TypeError, RecursionError) as e:
        a.exit(1, f"review refused: {e}\n")
