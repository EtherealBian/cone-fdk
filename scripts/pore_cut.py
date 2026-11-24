"""Development-only articulation ablation on archived C masks; see pores/ARTICULATION.md."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pores"))
from audit import decode  # noqa: E402
from pore_check import native, reference  # noqa: E402
from validate import ORIGIN, final_gates, objects, score, sha, stack, verified  # noqa: E402


def choose_cut(mask, minimum=64):
    """Iterative low-link DFS on a single 6-connected component (local ZYX coordinates)."""
    mask = np.pad(mask, 1)
    n = int(mask.sum())
    if n < 2 * minimum + 1:
        return None
    az, ay = mask.shape[1] * mask.shape[2], mask.shape[2]
    steps = (-az, -ay, -1, 1, ay, az)
    flat = mask.ravel()
    root = int(np.flatnonzero(flat)[0])
    parent = {root: -1}
    order = {root: 0}
    low = {root: 0}
    size = {root: 1}
    parts = {}
    work = [(root, 0)]
    best = None
    while work:
        v, d = work[-1]
        if d < 6:
            work[-1] = v, d + 1
            w = v + steps[d]
            if not flat[w] or w == parent[v]:
                continue
            if w in order:
                low[v] = min(low[v], order[w])
                continue
            order[w] = low[w] = len(order)
            parent[w] = v
            size[w] = 1
            work.append((w, 0))
            continue
        work.pop()
        sides = parts.pop(v, [])
        rest = n - 1 - sum(sides)
        if rest:
            sides.append(rest)
        if len(sides) == 2 and min(sides) >= minimum:
            choice = (min(sides), -v, max(sides))
            if best is None or choice > best:
                best = choice
        p = parent[v]
        if p != -1:
            size[p] += size[v]
            if low[v] >= order[p]:
                parts.setdefault(p, []).append(size[v])
            low[p] = min(low[p], low[v])
    if len(order) != n:
        raise ValueError("choose_cut expects exactly one connected component")
    if best is None:
        return None
    return tuple(int(a) - 1 for a in np.unravel_index(-best[1], mask.shape)), best[0], best[2]


def apply(pred):
    labels, _ = ndi.label(pred)
    out = pred.copy()
    cuts = []
    for label, box in enumerate(ndi.find_objects(labels), start=1):
        if any(b.start == 0 or b.stop == n for b, n in zip(box, pred.shape, strict=True)):
            continue
        piece = labels[box] == label
        cut = choose_cut(piece)
        if cut is not None:
            local, a, b = cut
            point = tuple(int(c + s.start) for c, s in zip(local, box, strict=True))
            z, y, x = point
            seed = tuple(
                j + s.start
                for j, s in zip(np.unravel_index(piece.argmax(), piece.shape), box, strict=True)
            )
            out[point] = False
            cuts.append(
                dict(
                    parent_id=int(np.ravel_multi_index(seed, pred.shape)) + 1,
                    source_x=x + ORIGIN[0],
                    source_y=y + ORIGIN[1],
                    source_z=z + ORIGIN[2],
                    daughter_voxels=[a, b],
                )
            )
    return out, cuts


def by_ref(path):
    with path.open(newline="") as f:
        return {int(row["ref_scipy_label"]): row for row in csv.DictReader(f)}


def run(folder, dest, cache, i):
    if i not in (2, 3, 5):
        raise ValueError("Development only")
    stem = f"sample{i}"
    report = json.loads((folder / (stem + ".json")).read_text(encoding="utf8"))
    review = folder / (stem + "-review.html")
    text = review.read_text(encoding="utf8")
    if (
        report["method"] != "multiscale boundary ensemble C"
        or report["specimen"] != i
        or sha(review) != report["review_sha256"]
        or report["origin_xyz"] != list(ORIGIN)
    ):
        raise ValueError("Expected the original candidate C review for this specimen")
    data = json.loads(
        re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S)[1]
    )
    _, ny, nx = data["shape"]
    pred = np.stack(
        [(decode(plane["objects"], ny * nx) != 0).reshape(ny, nx) for plane in data["layers"]]
    )
    del data, text
    mask, cuts = apply(pred)
    # Open the source labels for scoring only after all cut decisions are final.
    path, _ = verified(cache, i, "segmented", False)
    truth = stack(path, i, "segmented", 400, 528)
    if score(pred, truth) != report["metrics"]:
        raise ValueError("Unchanged control does not reproduce C")
    sp = report["native"]["spacing_xyz_um"]
    dest.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory() as tmp:
        table = Path(tmp) / "rows.csv"
        image = Path(tmp) / "mask.f32"
        (~mask).astype("<f4").tofile(image)
        st = native(image, table, mask.shape, sp, 0, 6, 37, "none", 64)
        lab, rows = reference(mask, 6, sp, table, st)
    new = dest / (stem + "-matches.csv")
    o = objects(lab, truth, rows, new)
    m = score(mask, truth)
    g = final_gates(m, o)
    before = by_ref(folder / (stem + "-matches.csv"))
    after = by_ref(new)
    changes = [
        dict(reference=r, before=before.get(r), after=after.get(r))
        for r in sorted(before.keys() | after.keys())
        if before.get(r) != after.get(r)
    ]
    record = dict(
        scope="development-only mask ablation of C, not a confirmation pipeline",
        specimen=i,
        source_review_sha256=sha(review),
        cut_count=len(cuts),
        cuts=cuts,
        changed_matches=changes,
        metrics=m,
        objects=o,
        gates=g,
    )
    (dest / (stem + ".json")).write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(
        i,
        "cuts",
        len(cuts),
        "changed matches",
        len(changes),
        "P95",
        o["p95_abs_volume_error_pct"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    a = parser.parse_args()
    for i in (2, 3, 5):
        run(a.source, a.out, a.cache, i)
