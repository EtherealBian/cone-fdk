"""Cross-threshold stability on an authenticated periodic campaign. Two trusted stdlib files only."""

import argparse
import hashlib
import math
import os
import stat
import sys
import types
from array import array
from collections import Counter, deque
from pathlib import Path

PROTOCOL = "f21f00ac180c0300b618feb2d5808615587ffe19529d8e0e54079a62db385890"
VERIFIER = "ca095a965c053210e03ed369c798f9dccc31bce968509f28b7c68d13fcbc7ece"
LIMITS = (4 << 20, 4 << 20, 4 << 20, 1 << 18, 1 << 20, 1 << 18)
MAX_CELLS = 2 << 20
MAX_DETAILS = 65536
CLASSES = (
    "always_flagged",
    "always_below",
    "fully_observed_mixed",
    "partially_observed",
    "always_unknown",
)
FOOT = ("all_flagged", "any_flagged", "any_flagged_or_unknown")


def load_verifier():
    p = Path(__file__).with_name("offline_campaign.py")
    with p.open("rb") as stream:
        source = stream.read((128 << 10) + 1)
    if hashlib.sha256(source).hexdigest() != VERIFIER:
        raise ValueError("unrecognized detached verifier source")
    # Only this independently pinned distributed Python module can execute, never an archive member.
    mod = types.ModuleType("pinned_detached_campaign")
    mod.__file__ = str(p)
    exec(compile(source, str(p), "exec"), mod.__dict__)
    return mod


c = load_verifier()
need = c.need


def labels(raw):
    out = array("i")
    out.frombytes(raw)
    need(out.itemsize == 4, "int32 required")
    if sys.byteorder != "little":
        out.byteswap()
    return out


def footprint(bits, h, w, anchors):
    n = h * w
    need(len(bits) == n and all(x in (0, 1) for x in bits), "binary footprint")
    pending = deque()
    assigned = array("i", [0]) * n
    areas = []
    for start in range(n):
        if not bits[start] or assigned[start]:
            continue
        root = start + 1
        assigned[start] = root
        pending.append(start)
        area = 0
        while pending:
            i = pending.popleft()
            area += 1
            for j in c.neighbours(i, w, n):
                if j >= 0 and bits[j] and not assigned[j]:
                    assigned[j] = root
                    pending.append(j)
        areas.append([root, area])
        need(len(areas) <= 32768, "footprint component limit")
    a, b = (assigned[y * w + x] for y, x in anchors)
    return dict(
        mask_sha256=c.sha(bytes(bits)),
        cells=sum(x[1] for x in areas),
        components=areas,
        anchor_ids=[a, b],
        anchors_linked=bool(a and a == b),
    )


def transition(left, right):
    m, a = left
    n, b = right
    need(len(m) == len(n) == len(a) == len(b), "different domains")
    grid = [[0] * 3 for _ in range(3)]
    overlap = Counter()
    lost = Counter()
    gained = Counter()
    src = Counter()
    dst = Counter()
    for u, v, i, j in zip(m, n, a, b, strict=True):
        grid[u][v] += 1
        if u == 1:
            src[i] += 1
        if v == 1:
            dst[j] += 1
        if u == v == 1:
            overlap[i, j] += 1
        elif u == 1:
            lost[i, v] += 1
        elif v == 1:
            gained[j, u] += 1
    out = Counter()
    inc = Counter()
    ns = Counter()
    nd = Counter()
    for (i, j), v in overlap.items():
        out[i] += v
        inc[j] += v
        ns[i] += 1
        nd[j] += 1
    for (i, _), v in lost.items():
        out[i] += v
    for (j, _), v in gained.items():
        inc[j] += v
    need(
        out == src
        and inc == dst
        and sum(overlap.values()) == grid[1][1]
        and sum(map(sum, grid)) == len(m),
        "overlap conservation",
    )

    def triples(d):
        return [[i, j, v] for (i, j), v in sorted(d.items())]

    return dict(
        support_matrix=grid,
        overlaps=triples(overlap),
        from_flagged_to_other=triples(lost),
        from_other_to_flagged=triples(gained),
        split_source_ids=sorted(i for i in src if ns[i] > 1),
        merged_target_ids=sorted(j for j in dst if nd[j] > 1),
        no_overlap_source_ids=sorted(i for i in src if not ns[i]),
        no_overlap_target_ids=sorted(j for j in dst if not nd[j]),
    )


def chart(frames, h, w, k, anchors, queries):
    keys = sorted(frames)
    need(
        keys == [(i, j) for i in range(k) for j in range(k)] and 1 <= k <= 3,
        "incomplete case product",
    )
    n = h * w
    count = k * k
    f = bytearray(n)
    u = bytearray(n)
    for key in keys:
        m, ls = frames[key]
        need(len(m) == len(ls) == n, "case size mismatch")
        for i, x in enumerate(m):
            need(
                x in (0, 1, 2) and (ls[i] > 0 if x == 1 else ls[i] == (0 if x == 0 else -1)),
                "mask/label mismatch",
            )
            f[i] += x == 1
            u[i] += x == 2
    votes = bytearray(2 * n)
    votes[0::2] = f
    votes[1::2] = u
    hist = Counter(zip(f, u))
    classes = dict.fromkeys(CLASSES, 0)
    masks = [bytearray(n) for _ in FOOT]
    for i, (yes, blind) in enumerate(zip(f, u, strict=True)):
        group = (
            4 if blind == count else 3 if blind else 0 if yes == count else 1 if not yes else 2
        )
        classes[CLASSES[group]] += 1
        masks[0][i] = yes == count
        masks[1][i] = yes != 0
        masks[2][i] = yes + blind != 0
    footprints = {
        name: footprint(bits, h, w, anchors) for name, bits in zip(FOOT, masks, strict=True)
    }
    linked = []
    for key in keys:
        la = frames[key][1]
        a, b = [la[y * w + x] for y, x in anchors]
        if a > 0 and a == b:
            linked.append(list(key))
    edges = []
    units = sum(len(x["components"]) for x in footprints.values())
    for i, j in keys:
        for dest in ((i + 1, j), (i, j + 1)):
            if dest not in frames:
                continue
            t = transition(frames[i, j], frames[dest])
            units += sum(len(x) for x in t.values())
            need(units <= MAX_DETAILS, "per-case ledger budget")
            edges.append(dict(source=[i, j], target=list(dest), **t))
    points = []
    for y, x in sorted(queries):
        need(c.integer(y, 0, h - 1) and c.integer(x, 0, w - 1), "point outside source grid")
        i = y * w + x
        points.append(
            dict(
                row_col=[y, x],
                flagged_count=f[i],
                unknown_count=u[i],
                labels_by_pair=[frames[k][1][i] for k in keys],
            )
        )
    return dict(
        shape_ha=[h, w],
        ordered_pairs=[list(k) for k in keys],
        votes_sha256=c.sha(votes),
        vote_counts=[[a, b, n] for (a, b), n in sorted(hist.items())],
        partition=classes,
        footprints=footprints,
        observed_link_settings=linked,
        union_link_without_single_setting=footprints["any_flagged"]["anchors_linked"]
        and not linked,
        transitions=edges,
        points=points,
    ), units


def atlas(stream, root, points=()):
    r = c.Reader(stream)
    chain = c.obj(r.get(c.pin(root), c.MAX_DOC))
    survey = c.obj(r.get(chain["spark_survey_sha256"], c.MAX_DOC))
    grid, refused = c.plan(chain, survey)
    need(chain["cells"] <= MAX_CELLS, "atlas cell budget")
    need(
        len(points) <= 64
        and len(set(points)) == len(points)
        and all(name in survey["cases"] and name not in refused for name, _, _ in points),
        "unknown, withheld or duplicate point query",
    )
    groups = {}
    out = {}
    cells = units = 0
    for row in chain["rows"]:
        name, i, j = c.ident(row)
        parts = [r.get(row[slot], limit) for slot, limit in zip(c.SLOTS, LIMITS, strict=True)]
        p, table = c.bind(chain, row, parts)
        c.audit(*parts[3:], p, table, grid[name, i, j])
        cells += len(parts[3])
        need(cells <= MAX_CELLS, "atlas cell budget")
        h, w = p["shape_ha"]
        levels = p["levels_per_mm"]
        k = len(levels)
        need(
            1 <= k <= 3
            and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in levels)
            and sorted(set(levels)) == levels
            and max(i, j) < k
            and p["pair_levels_per_mm"] == [levels[i], levels[j]],
            "threshold value binding",
        )
        meta = {
            s: p[s]
            for s in (
                "shape_ha",
                "anchors_row_col",
                "layout",
                "edge_manifest_sha256",
                "material_grid_sha256",
                "levels_per_mm",
                "loss_gate_mm",
            )
        }
        c.pin(meta["material_grid_sha256"])
        need(
            meta["layout"]
            == "scenario selected by pair; rows axial, columns azimuth; only columns wrap; i32 little-endian: -1 unknown, 0 below, positive first row-major cell + 1",
            "unrecognized grid layout",
        )
        gate = meta["loss_gate_mm"]
        need(type(gate) in (int, float) and math.isfinite(gate) and gate > 0, "invalid loss gate")
        need(len(survey["cases"][name]["rows"]) == k * k, "incomplete level set")
        base, frames, parents = groups.setdefault(name, (meta, {}, {}))
        need(
            c.encoded(base) == c.encoded(meta) and (i, j) not in frames and name not in out,
            "cross-setting case identity",
        )
        frames[i, j] = parts[3], labels(parts[4])
        parents[i, j] = row[c.SLOTS[0]]
        if len(frames) == k * k:
            result, added = chart(
                frames,
                h,
                w,
                k,
                meta["anchors_row_col"],
                [(y, x) for nm, y, x in points if nm == name],
            )
            units += added
            need(units <= MAX_DETAILS, "campaign ledger budget")
            out[name] = dict(
                source=meta,
                parcel_sha256_by_pair=[parents[key] for key in sorted(parents)],
                **result,
            )
            del groups[name]
    need(
        not groups
        and cells == chain["cells"]
        and set(out) | set(refused) == set(survey["cases"])
        and not stream.read(1),
        "incomplete atlas or trailing data",
    )
    return dict(
        format="CFDK-verified-threshold-sensitivity-atlas-v1",
        protocol_sha256=PROTOCOL,
        detached_verifier_sha256=VERIFIER,
        source_chain_sha256=root,
        source_recipe_sha256=chain["source_recipe_sha256"],
        maps=chain["maps"],
        scenario_cells=cells,
        cases=out,
        upstream_refused=refused,
        scope="Observed overlap on the sampled grids of the supplied cases. Votes are counts of settings, not probabilities; an overlap is not a physical identity. Pooled paths need not exist for any single setting. This campaign is not physical specimen metrology.",
    )


def review(path, root, points=()):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as f:
        s = os.fstat(f.fileno())
        need(
            stat.S_ISREG(s.st_mode) and 0 < s.st_size <= c.MAX_PACK,
            "expected bounded regular pack",
        )
        return atlas(f, root, points)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pack", type=Path)
    cli.add_argument("source_chain_sha256")
    cli.add_argument(
        "--point", nargs=3, action="append", default=[], metavar=("CASE", "ROW", "COLUMN")
    )
    args = cli.parse_args()
    try:
        doc = review(
            args.pack,
            args.source_chain_sha256,
            [(case, int(y), int(x)) for case, y, x in args.point],
        )
        sys.stdout.buffer.write(c.encoded(doc) + b"\n")
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        IndexError,
        OverflowError,
        RecursionError,
    ) as e:
        cli.exit(1, f"atlas refused: {e}\n")


if __name__ == "__main__":
    main()
