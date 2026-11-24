"""Sealed, reference-free minimum vertex cut for exactly two bridge cores."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_bridge_review import flag
from pore_transfer import DEV, ROOT, digest, sha
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

Z = (552, 680)
SHAPE = (128, 256, 256)
SEALED = ROOT / "pores/evidence/split-pilot/seal.json"


def vertex_cut(part, p, cores, a, b):
    """Node splitting, cut verified against primal max flow; source-minimal tie handling."""
    n = int(part.sum())
    ids = np.full(part.shape, -1, np.int32)
    ids[part] = np.arange(n, dtype=np.int32)
    k = cores[part]
    cost = 1 + np.floor(100 * (p[part].astype("f8") - 0.5)).astype("i8")
    free = (k != a) & (k != b)
    inf = int(cost[free].sum()) + 1
    if inf > np.iinfo(np.int32).max:
        raise OverflowError("Graph exceeds the CPU flow backend capacity")
    cost[~free] = inf
    v = ids[part]
    src, sink = 2 * n, 2 * n + 1
    fr = [2 * v]
    to = [2 * v + 1]
    weights = [cost]

    def edge(x, y):
        fr.append(x)
        to.append(y)
        weights.append(np.full(len(x), inf, dtype="i8"))

    for dim in range(3):
        u, w = np.moveaxis(ids, dim, 0)[:-1], np.moveaxis(ids, dim, 0)[1:]
        ok = (u >= 0) & (w >= 0)
        lhs, rhs = u[ok], w[ok]
        edge(2 * lhs + 1, 2 * rhs)
        edge(2 * rhs + 1, 2 * lhs)
    s, t = v[k == a], v[k == b]
    edge(np.full(len(s), src), 2 * s)
    edge(2 * t + 1, np.full(len(t), sink))
    cap = coo_matrix(
        (np.concatenate(weights), (np.concatenate(fr), np.concatenate(to))),
        shape=(sink + 1, sink + 1),
    ).tocsr()
    f = maximum_flow(cap, src, sink)
    residual = cap - f.flow
    residual.eliminate_zeros()
    reached = np.zeros(sink + 1, bool)
    reached[breadth_first_order(residual, src, return_predecessors=False)] = True
    removed = reached[2 * v] & ~reached[2 * v + 1]
    if reached[sink] or np.any(removed & ~free) or cost[removed].sum() != f.flow_value:
        raise ValueError("Invalid minimum-cut certificate")
    cut = np.zeros_like(part)
    cut[part] = removed
    return cut, int(f.flow_value)


def split(p):
    labs, rows = flag(p)
    mask = labs > 0
    boxes = ndi.find_objects(labs)
    for row in rows:
        row.update(decision="not flagged", removed_linear=[], daughter_seeds=[])
        if not row["flagged"]:
            continue
        if len(row["core_sizes"]) != 2:
            row["decision"] = "more than two cores"
            continue
        pid = row["pred"]
        box = boxes[pid - 1]
        part = labs[box] == pid
        cores, _ = ndi.label(part & (p[box] >= 0.90))
        sizes = np.bincount(cores.ravel())
        sizes[0] = 0
        a, b = np.flatnonzero((sizes >= 32) & (20 * sizes >= row["voxels"]))
        cut, val = vertex_cut(part, p[box], cores, a, b)
        daughters, n = ndi.label(part & ~cut)
        ds = np.bincount(daughters.ravel(), minlength=n + 1)[1:]
        row.update(
            cut_cost=val, proposed_voxels=int(cut.sum()), proposed_daughter_volumes=ds.tolist()
        )
        if int(cut.sum()) * 50 > row["voxels"]:
            row["decision"] = "cut exceeds 2%"
            continue
        which = [np.unique(daughters[cores == cid]).tolist() for cid in (a, b)]
        if n != 2 or ds.min() < 64 or sorted(which) != [[1], [2]]:
            row["decision"] = "ineligible daughters"
            continue

        def global_indices(region, box=box):
            return np.ravel_multi_index(
                tuple(q + sl.start for q, sl in zip(np.nonzero(region), box, strict=True)),
                p.shape,
            )

        mask[box][cut] = False
        row.update(
            decision="edited",
            removed_linear=global_indices(cut).tolist(),
            daughter_seeds=[int(global_indices(daughters == j)[0]) for j in (1, 2)],
        )
    return mask, rows


def load_prob(folder, manifest, i):
    pub = json.loads((ROOT / f"pores/evidence/candidate-e/sample{i}.json").read_text())
    r = manifest["specimens"][str(i)]
    if (
        i not in DEV
        or set(manifest["specimens"]) != {str(k) for k in DEV}
        or manifest["model"] != pub["pipeline_id"]
        or manifest["z"] != list(Z)
        or manifest["shape_zyx"] != list(SHAPE)
        or manifest["dtype"] != "<f4"
        or any(r[k] != pub["source"][k]["sha256"] for k in ("raw", "segmented"))
    ):
        raise ValueError("Unregistered population")
    path = folder / f"sample{i}.f32"
    if sha(path) != r["probability"] or path.stat().st_size != 4 * np.prod(SHAPE):
        raise ValueError("Wrong probability volume")
    return np.memmap(path, dtype="<f4", mode="r", shape=SHAPE)


def seal(folder):
    manifest = json.loads((folder / "baseline.json").read_text())
    arms = {}
    for i in DEV:
        p = load_prob(folder, manifest, i)
        mask, rows = split(p)
        arms[str(i)] = dict(
            base=digest(p >= 0.5),
            arm=digest(mask),
            eligible=len(rows),
            flagged=sum(r["flagged"] for r in rows),
            edited=sum(r["decision"] == "edited" for r in rows),
        )
    return dict(baseline=manifest, arms=arms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = seal(args.folder)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf8") as fp:
        json.dump(data, fp, indent=2)
        fp.write("\n")
