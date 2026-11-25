"""Connected wall-loss footprints under a single edge-level pair for the entire specimen."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from edge_review import CLASSES, MAPS, ROLES, check, scenarios, thickness
from remount_loss import write_doc
from scipy import ndimage
from wall import WALL_DTYPE

# Explicit offline working-set limits: refuse larger jobs, never truncate a component list.
MAX_CELLS = 1 << 20
MAX_PAIR_CELLS = 32 << 20
MAX_WALL_CELLS = 8 << 20
MAX_DETAILS = 65536  # component + overlap records across all scenarios


def read_bound(path, limit, expected=None, exact=None):
    with path.open("rb") as f:
        size = f.seek(0, 2)
        check(size <= limit and (exact is None or size == exact), "input size: " + path.name)
        f.seek(0)
        data = f.read(size + 1)
    check(len(data) == size, "input changed size: " + path.name)
    digest = hashlib.sha256(data).hexdigest()
    check(expected is None or digest == expected, "SHA-256 mismatch: " + path.name)
    return data, digest


def grid_header(data):
    tok = data.decode("utf-8-sig").split()
    check(len(tok) == 16 and tok[0] == "CFDKCYL1", "invalid grid header")
    w, h, k = map(int, tok[1:4])
    frame = np.array(tok[4:], dtype=float).reshape(4, 3)
    _, axis, seam, (r0, dr, dz) = frame
    check(
        3 <= w <= 4096
        and 1 <= h <= 4096
        and 3 <= k <= 4096
        and np.isfinite(frame).all()
        and min(r0, dr, dz) > 0
        and max(abs(np.linalg.norm(axis) - 1), abs(np.linalg.norm(seam) - 1), abs(axis @ seam))
        <= 1e-6,
        "invalid material grid",
    )
    return (h, w), float(r0), float(r0 + (k - 1) * dr), float(dz)


def load(job, expected):
    raw, digest = read_bound(job / "manifest.json", 1 << 20, expected)
    doc = json.loads(raw)
    check(
        doc["format"] == "CFDK-edge-review-v1" and doc["class_codes"] == list(CLASSES),
        "not an edge review",
    )
    levels = scenarios(doc["levels_per_mm"], doc["nominal_level_per_mm"])
    gate = doc["loss_gate_mm"]
    check(
        levels == doc["levels_per_mm"]
        and doc["independent_pairs"] == len(levels) ** 2
        and math.isfinite(gate)
        and gate > 0,
        "invalid pair metadata",
    )
    hashes = doc["outputs_sha256"]

    def artifact(name, cap, exact=None):
        check(isinstance(hashes[name], str), "missing SHA-256: " + name)
        return read_bound(job / name, cap, hashes[name], exact)[0]

    shape, lo, hi, dz = grid_header(artifact("material.grid", 4096))
    n, k = math.prod(shape), len(levels)
    check(list(shape) == doc["shape_ha"] and math.isfinite(hi) and hi > lo, "inconsistent grid")
    check(
        n <= MAX_CELLS and n * k * k <= MAX_PAIR_CELLS and 2 * n * k <= MAX_WALL_CELLS,
        "offline review size limit",
    )
    walls = []
    for role in ROLES:
        group = []
        for i in range(k):
            raw = artifact(f"{role}-{i:02}.wall", n * 32, n * 32)
            group.append(thickness(np.frombuffer(raw, WALL_DTYPE), lo, hi).reshape(shape).copy())
        walls.append(group)
    maps = [
        np.frombuffer(artifact(name, n * s, n * s), dtype).reshape(shape)
        for name, dtype, s in zip(MAPS, ("u1", "<f8", "<f8"), (1, 8, 8), strict=True)
    ]
    return doc, digest, walls, maps, (lo, hi, dz)


def labels(mask):
    """4-connected components on a cylinder, ID = first row-major cell + 1."""
    lab, count = ndimage.label(mask)  # planar 4-neighbours
    parents = np.arange(count + 1, dtype=np.int32)

    def root(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for a, b in zip(lab[:, 0], lab[:, -1], strict=True):
        if a and b and a != b:
            x, y = root(a), root(b)
            parents[max(x, y)] = min(x, y)
    for i in range(1, count + 1):
        parents[i] = parents[parents[i]]  # parents always smaller, already resolved
    merged = parents[lab]
    active = np.flatnonzero(mask)
    first = np.full(count + 1, mask.size + 1, np.int32)
    np.minimum.at(first, merged.ravel()[active], active + 1)
    first[0] = 0
    return first[merged]


def scenario(delta, gate, nominal, cell_area, budget):
    hit, unknown = delta >= gate, np.isnan(delta)
    lab = labels(hit)
    h, w = lab.shape
    sites = np.flatnonzero(hit)
    ids, compact, counts = np.unique(lab.ravel()[sites], return_inverse=True, return_counts=True)
    check(len(ids) <= budget, "component detail limit")
    peaks = np.full(len(ids), -np.inf)
    values = delta.ravel()[sites]
    np.maximum.at(peaks, compact, values)
    at_peak = values == peaks[compact]
    peak_sites = np.full(len(ids), lab.size, np.int32)
    np.minimum.at(peak_sites, compact[at_peak], sites[at_peak])
    seam_ids = set(lab[:, 0][hit[:, 0] & hit[:, -1]])
    # Outside the sampled axial rows is not a measured neighbour; report it separately.
    rim = np.roll(unknown, 1, axis=1) | np.roll(unknown, -1, axis=1)
    rim[1:] |= unknown[:-1]
    rim[:-1] |= unknown[1:]
    blind_ids = set(np.unique(lab[rim & hit]))
    end_ids = set(np.unique(lab[[0, -1]]))
    regs = [
        dict(
            id=int(r),
            cells=int(c),
            area_mm2=float(c * cell_area),
            peak_loss_mm=float(p),
            peak_cell=list(map(int, divmod(int(at), w))),
            crosses_seam=r in seam_ids,
            adjacent_unknown=r in blind_ids,
            touches_axial_end=r in end_ids,
        )
        for r, c, p, at in zip(ids, counts, peaks, peak_sites, strict=True)
    ]
    # Each edge is an actual overlap in material coordinates, not a nearest-centre match.
    same = hit & (nominal > 0)
    base = h * w + 1
    keys = nominal[same].astype("i8") * base + lab[same]
    pairs, sizes = np.unique(keys, return_counts=True)
    check(len(ids) + len(pairs) <= budget, "overlap detail limit")
    overlap = [[int(v // base), int(v % base), int(c)] for v, c in zip(pairs, sizes, strict=True)]
    # Zero is below, -1 is unknown, positive IDs distinguish the actual components.
    lab[unknown] = -1
    return dict(
        observed_flagged_cells=len(sites),
        unknown_cells=int(unknown.sum()),
        region_count=len(regs),
        largest_observed_cells=int(counts.max(initial=0)),
        regions=regs,
        nominal_overlap=overlap,
        labels_i32le_sha256=hashlib.sha256(lab.astype("<i4", copy=False).tobytes()).hexdigest(),
    ), lab


def extrema(cases, key):
    if not cases:
        return None
    out = {}
    for name, op in [("min", min), ("max", max)]:
        value = op(s[key] for s in cases)
        out[name] = dict(value=value, pairs=[s["pair"] for s in cases if s[key] == value])
    return out


def analyze(review, area_radius_mm, review_sha256=None):
    doc, digest, (ref, part), (classes, lower, upper), (lo, hi, dz) = load(
        Path(review), review_sha256
    )
    levels, gate = doc["levels_per_mm"], doc["loss_gate_mm"]
    check(
        math.isfinite(area_radius_mm) and lo <= area_radius_mm <= hi,
        "area radius outside sampled shells",
    )
    h, w = classes.shape
    area = math.tau * area_radius_mm / w * dz
    check(math.isfinite(area * h * w) and area > 0, "invalid projected area")
    k = levels.index(doc["nominal_level_per_mm"])
    nominal_delta = ref[k] - part[k]
    nominal = labels(nominal_delta >= gate)
    nominal_tag = np.where(np.isnan(nominal_delta), 2, nominal > 0)
    mins = np.full((h, w), np.inf)
    maxes = -mins
    observed_union = np.zeros((h, w), bool)
    budget = MAX_DETAILS
    cases = []
    for i, a in enumerate(ref):
        for j, b in enumerate(part):
            delta = a - b
            out, lab = scenario(delta, gate, nominal, area, budget)
            cases.append(dict(pair=[i, j], **out))
            budget -= out["region_count"] + len(out["nominal_overlap"])
            np.minimum(mins, delta, out=mins)
            np.maximum(maxes, delta, out=maxes)
            observed_union |= lab > 0
    code = np.select([np.isnan(mins), mins >= gate, maxes >= gate], [3, 2, 1], default=0).astype(
        "u1"
    )
    check(
        np.array_equal(code, classes)
        and np.array_equal(mins, lower, equal_nan=True)
        and np.array_equal(maxes, upper, equal_nan=True),
        "explicit pairs disagree with stored cell envelope",
    )
    table = np.bincount((4 * nominal_tag + code).ravel(), minlength=12).reshape(3, 4)
    counts = dict(zip(CLASSES, table.sum(0).tolist(), strict=True))
    cross = {
        name: dict(zip(CLASSES, row.tolist(), strict=True))
        for name, row in zip(("below", "flagged", "unknown"), table, strict=True)
    }
    check(counts == doc["counts"] and cross == doc["nominal_by_class"], "cell counts disagree")
    return dict(
        format="CFDK-region-review-v1",
        edge_manifest_sha256=digest,
        shape_ha=[h, w],
        levels_per_mm=levels,
        nominal_pair=[k, k],
        loss_gate_mm=gate,
        area_radius_mm=area_radius_mm,
        projected_cell_area_mm2=area,
        connectivity=4,
        periodic_columns=True,
        observed_union_cells=int(observed_union.sum()),
        always_flagged_cells=counts["always_flagged"],
        incomplete_pair_count=sum(s["unknown_cells"] != 0 for s in cases),
        observed_extrema={
            key: extrema(cases, key)
            for key in ("observed_flagged_cells", "region_count", "largest_observed_cells")
        },
        complete_grid_flagged_extrema=extrema(
            [s for s in cases if not s["unknown_cells"]], "observed_flagged_cells"
        ),
        scenarios=cases,
    )


def run(review, output, area_radius_mm, review_sha256=None):
    review, output = Path(review).resolve(), Path(output).resolve()
    check(not output.is_relative_to(review), "write the report outside its input")
    result = analyze(review, area_radius_mm, review_sha256)
    output.mkdir(parents=True, exist_ok=False)
    tmp = output / "manifest.partial.json"
    write_doc(tmp, result)
    tmp.replace(output / "manifest.json")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("review", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--area-radius-mm", type=float, required=True)
    p.add_argument(
        "--review-sha256", help="edge manifest digest from a trusted independent record"
    )
    try:
        result = run(**vars(p.parse_args()))
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as e:
        p.exit(1, f"region review failed: {e}\n")
    print(json.dumps(result["observed_extrema"]))
    raise SystemExit(2 if result["incomplete_pair_count"] else 0)
