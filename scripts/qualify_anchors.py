"""Verify portable anchor certificates against independent pinned CUDA runs and a heap oracle."""

import argparse
import hashlib
import heapq
import json
import struct
from pathlib import Path

from anchor_audit import audit, publish


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def unpack(p, expected):
    raw = p.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "native file digest")
    m = json.loads(raw)
    h, w = m["rows"], m["columns"]
    lab = [[0] * w for _ in range(h)]
    for r, c, n, v in m["runs"]:
        require(
            0 <= r < h and 0 <= c < c + n <= w and v != 0 and not any(lab[r][c : c + n]),
            "native span",
        )
        lab[r][c : c + n] = [max(-1, v)] * n
    return lab


def oracle(lab, a, b):
    """Lexicographic Dijkstra on material coordinates, independently decoded native runs."""
    h, w = len(lab), len(lab[0])
    sy, sx = a
    ty, tx = b
    if 0 in (lab[sy][sx], lab[ty][tx]):
        return "anchor_below_gate", None, None
    heap = [(int(lab[sy][sx] < 0), 1, sy, sx)]
    closed = set()
    while heap:
        cost, cells, y, x = heapq.heappop(heap)
        if (y, x) in closed:
            continue
        if (y, x) == (ty, tx):
            return ("unresolved_blind_path" if cost else "observed_link"), cost, cells
        closed.add((y, x))
        for yy, xx in ((y, (x - 1) % w), (y, (x + 1) % w), (y - 1, x), (y + 1, x)):
            if 0 <= yy < h and lab[yy][xx] and (yy, xx) not in closed:
                heapq.heappush(heap, (cost + int(lab[yy][xx] < 0), cells + 1, yy, xx))
    return "separated_on_grid", None, None


def check_row(r, lab, anchors):
    a, b = anchors
    h, w = len(lab), len(lab[0])
    sig = hashlib.sha256(b"".join(struct.pack("<" + "i" * w, *row) for row in lab)).hexdigest()
    require(sig == r["labels_i32le_sha256"], "unbound native map")
    require(
        r["anchor_label_ids"] == [lab[y][x] for y, x in anchors],
        "anchor labels differ from native",
    )
    state, cost, length = oracle(lab, a, b)
    path = r["path"]
    require(
        state == r["state"]
        and cost == r["min_unknown_cells"]
        and length == (path and path["cells"]),
        "oracle cost, length or state differs",
    )
    if path is not None:
        y, x = a
        vertices = [y * w + x]
        blind = int(lab[y][x] < 0)
        for heading, n in path["direction_runs"]:
            dy, dx = {"E": (0, 1), "W": (0, -1), "N": (-1, 0), "S": (1, 0)}[heading]
            require(type(n) is int and 0 < n <= h * w, "bad certificate run")
            for _ in range(n):
                y, x = y + dy, (x + dx) % w
                require(
                    0 <= y < h and lab[y][x] != 0,
                    "path traverses unmeasured boundary or below-gate cell",
                )
                blind += int(lab[y][x] < 0)
                vertices.append(y * w + x)
        require(
            [y, x] == b
            and len(vertices) == len(set(vertices)) == length
            and blind == cost
            and hashlib.sha256(struct.pack("<" + "i" * length, *vertices)).hexdigest()
            == path["row_major_i32le_sha256"],
            "invalid witness path",
        )


def qualify(reports, bridge_native, legacy_native, out):
    require(not out.exists(), "fresh work required")
    out.mkdir(parents=True)
    study = json.loads((reports / "bridge-result.json").read_bytes())
    old = json.loads((reports / "region-qualification.json").read_bytes())
    bb = json.loads((reports / "bridge-bundle.json").read_bytes())
    ref = json.loads((reports / "region-console-browser.json").read_bytes())
    require(
        bb["measured_result_sha256"] == digest(reports / "bridge-result.json")
        and ref["native_qualification_sha256"] == digest(reports / "region-qualification.json"),
        "source record mismatch",
    )
    cases = {}
    n = 0
    cells = 0
    for name in [*study["cases"], "invalid-edge-control"]:
        legacy = name == "invalid-edge-control"
        record = (old if legacy else study)["cases"][name]
        root = (legacy_native if legacy else bridge_native) / name
        page = reports / ("region-console.html" if legacy else "bridge-console.html")
        pin = ref["html_sha256"] if legacy else bb["console"]["html_sha256"]
        anchors = [[11, 223], [12, 218]] if legacy else [[11, 46], [29, 46]]
        d = audit(page, pin, name, anchors)
        require(
            d["edge_sha256"] == record["edge_sha256"]
            and d["region_sha256"] == record["region_sha256"],
            "incorrect source lineage",
        )
        rows = {tuple(s["pair"]): s for s in record["cases" if legacy else "rows"]}
        require(len(rows) == len(d["scenarios"]), "missing scenarios")
        for r in d["scenarios"]:
            i, j = r["pair"]
            x = rows[i, j]
            p = root / f"native-{i}-{j}.json"
            lab = unpack(p, x["native_report_sha256" if legacy else "native_sha256"])
            check_row(r, lab, anchors)
            n += 1
            cells += len(lab) * len(lab[0])
        artifact = out / (name + ".json")
        publish(artifact, d)
        cases[name] = dict(counts=d["counts"], audit_sha256=digest(artifact))
    require(n == 40 and cells == 590400, "scope")
    publish(
        out / "qualification.json",
        dict(
            scope="40 portable maps independently decoded from CUDA runs. Lexicographic heap oracle and replay of every path witness; incomplete paths remain hypothetical.",
            source_sha256={
                f: digest(Path(__file__).with_name(f))
                for f in ("anchor_audit.py", "qualify_anchors.py")
            },
            maps=n,
            pair_cells=cells,
            cases=cases,
        ),
    )
    print("PASS:", n, "maps;", cells, "native label cells")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("reports", "bridge_native", "legacy_native", "out"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
