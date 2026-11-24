"""Compare deposited digital pore masks in fixed external crops; not a ConeFDK score."""

import argparse
import hashlib
import json
import zipfile
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np
from pore_external_slice_inventory import API, DOI, SAMPLES, cached
from scipy import ndimage as ndi

METHODS = ("Binary", "Adaptive", "UNet")


def bounds(shape):
    z, y, x = map(int, shape)
    if min(shape) <= 0 or z < 256:
        raise ValueError("Volume too small for fixed crop")
    # Half the length in each transverse dimension, rounding size down, centred.
    sizes = (256, y // 2, x // 2)
    return [[(n - m) // 2, (n - m) // 2 + m] for n, m in zip((z, y, x), sizes, strict=True)]


def decode(plane, pore):
    if pore == 255:
        if np.any((plane != 0) & (plane != 255)):
            raise ValueError("Unexpected binary code")
    elif pore != 1 or plane.max() > 3:
        raise ValueError("Unexpected multiclass code")
    return plane == pore


def polarity(path, human, pore):
    # The authors' test indices address the deposited segmentation, not the uncropped NIST scan.
    with np.load(human, allow_pickle=False) as h:
        z = h["z"]
        ref = h["slices"]
        codes = tuple(int(h[k].item()) for k in ("pore", "metal", "background"))
    if (
        codes != (1, 2, 3)
        or z.ndim != 1
        or z.dtype.kind not in "iu"
        or len(z) != 4
        or len(set(z.tolist())) != 4
        or ref.ndim != 3
        or len(ref) != 4
    ):
        raise ValueError("Invalid primary annotation")
    with zipfile.ZipFile(path) as arc, arc.open("volume.npy") as f:
        if np.lib.format.read_magic(f) != (1, 0):
            raise ValueError("Unexpected array version")
        (nz, ny, nx), order, dt = np.lib.format.read_array_header_1_0(f)
        offset = f.tell()
        if (
            order
            or dt != np.dtype("u1")
            or ref.shape[1:] != (ny, nx)
            or min(z) < 0
            or max(z) >= nz
        ):
            raise ValueError("Incorrect test grid")
        rows = []
        for idx in np.argsort(z):
            zi = int(z[idx])
            f.seek(offset + zi * ny * nx)
            plane = np.frombuffer(f.read(ny * nx), dtype=dt).reshape(ny, nx)
            pred = decode(plane, pore)
            hp, metal = ref[idx] == 1, ref[idx] == 2
            rows.append(
                [
                    zi,
                    int(hp.sum()),
                    int(metal.sum()),
                    int(np.count_nonzero(pred & hp)),
                    int(np.count_nonzero(pred & metal)),
                ]
            )
    _, nh, nm, tp, fp = np.sum(rows, axis=0).tolist()
    if not (tp and nm and tp * nm > fp * nh):
        raise ValueError(
            "Foreground not enriched on primary human pore pixels; do not interpret volumes"
        )
    return rows


def read_mask(path, pore, roi=None):
    with np.load(path, allow_pickle=False) as f:
        shape = f["shape"].tolist()
        spacing = f["spacing"].tolist()
    with zipfile.ZipFile(path) as archive, archive.open("volume.npy") as f:
        if np.lib.format.read_magic(f) != (1, 0):
            raise ValueError("Unexpected array version")
        dims, order, dt = np.lib.format.read_array_header_1_0(f)
        if (
            list(dims) != shape
            or len(dims) != 3
            or dt != np.dtype("u1")
            or order
            or np.shape(spacing) != (3,)
            or not np.all(np.isfinite(spacing))
            or min(spacing) <= 0
            or archive.getinfo("volume.npy").file_size != f.tell() + int(np.prod(dims))
        ):
            raise ValueError("Unexpected volume geometry/type")
        roi = bounds(shape) if roi is None else roi
        if len(roi) != 3 or not all(0 <= a < b <= n for (a, b), n in zip(roi, dims, strict=True)):
            raise ValueError("Invalid bounds")
        (z0, z1), (y0, y1), (x0, x1) = roi
        _, ny, nx = dims
        # Bound memory even when the crop starts >1 GiB into a compressed member.
        skip = z0 * ny * nx
        while skip:
            block = f.read(min(8 << 20, skip))
            if not block:
                raise ValueError("Truncated archive")
            skip -= len(block)
        mask = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=bool)
        for a in range(z0, z1, 8):
            n = min(8, z1 - a)
            plane = np.frombuffer(f.read(n * ny * nx), dtype=dt).reshape(n, ny, nx)[
                :, y0:y1, x0:x1
            ]
            mask[a - z0 : a - z0 + n] = decode(plane, pore)
    return mask, dict(shape_zyx=shape, spacing_as_stored=spacing, crop_zyx=roi)


def segment(mask):
    labels, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    where = np.flatnonzero(mask)
    ids, first, counts = np.unique(labels.ravel()[where], return_index=True, return_counts=True)
    edges = set(
        np.unique(
            np.concatenate(
                [np.take(labels, i, axis=a).ravel() for a in range(3) for i in (0, -1)]
            )
        ).tolist()
    )
    eligible = {
        int(k): [int(v), list(map(int, np.unravel_index(where[f], mask.shape)))]
        for k, f, v in zip(ids, first, counts, strict=True)
        if k not in edges and v >= 64
    }
    return labels, dict(total_components=int(n), pore_voxels=int(len(where)), eligible=eligible)


def compare(a, b, ac, bc):
    if a.shape != b.shape:
        raise ValueError("Grid mismatch")
    both = (a != 0) & (b != 0)
    stride = int(b.max()) + 1
    keys, intersections = np.unique(a[both].astype("i8") * stride + b[both], return_counts=True)
    choices = []
    for key, v in zip(keys, intersections, strict=True):
        i, j = divmod(int(key), stride)
        v = int(v)
        if i in ac and j in bc:
            union = ac[i][0] + bc[j][0] - v
            if 2 * v >= union:
                choices.append((Fraction(v, union), i, j, v))
    left, right = set(), set()
    rows = []
    deltas = []
    disjoint = 0
    for _iou, i, j, v in sorted(choices, reverse=True):
        if i in left or j in right:
            continue
        left.add(i)
        right.add(j)
        va, vb = ac[i][0], bc[j][0]
        rows.append([i, j, v])
        deltas.append(100 * abs(vb - va) / va)
        disjoint += 115 * min(va, vb) < 85 * max(va, vb)
    return dict(
        matches=rows,
        unmatched_a=sorted(ac.keys() - left),
        unmatched_b=sorted(bc.keys() - right),
        p95_abs_difference_pct_wrt_a=float(np.percentile(deltas, 95)) if deltas else None,
        b_smaller=sum(bc[j][0] < ac[i][0] for i, j, _ in rows),
        b_larger=sum(bc[j][0] > ac[i][0] for i, j, _ in rows),
        no_common_estimate_within_15pct_of_both=disjoint,
    )


def verified(cache, record, folder, name):
    files = [
        f
        for f in record["files"]
        if record["folder_structure"][str(f["id"])] == folder and f["name"] == name
    ]
    if len(files) != 1:
        raise ValueError(f"Ambiguous file: {folder}/{name}")
    f = files[0]
    path = cache / f"{f['id']}-{name}"
    blob = cached(path, f["download_url"])
    if len(blob) != f["size"] or hashlib.md5(blob).hexdigest() != f["computed_md5"]:
        raise ValueError("Published integrity check failed")
    return path, dict(file_id=f["id"], sha256=hashlib.sha256(blob).hexdigest())


def run(cache):
    cache.mkdir(exist_ok=True, parents=True)
    record = json.loads(cached(cache / "record-v1.json", API))
    if record["doi"] != DOI:
        raise ValueError("Unexpected record")
    path, classes = verified(cache, record, "Segmentation", "class_labels.json")
    schema = json.loads(path.read_bytes())
    if schema != {"unlabeled": 0, "pore": 1, "metal": 2, "background": 3}:
        raise ValueError("Unexpected class mapping")
    out = dict(
        doi=DOI,
        scope="method-to-method only; no grayscale, human 3D reference, or ConeFDK inference",
        class_file=classes,
        polarity_rows="[deposited_z, primary_pore, primary_metal, foreground_on_primary_pore, foreground_on_primary_metal]",
        eligible_values="[voxels, seed_zyx_relative_to_crop]",
        match_rows="[a_id,b_id,intersection_voxels]",
        specimens={},
    )
    for sample in SAMPLES:
        volumes = {}
        methods = {}
        geometry = None
        human, hsrc = verified(cache, record, "Test_Data/Primary", sample + "_slices.npz")
        for method in METHODS:
            file, source = verified(cache, record, "Segmentation/" + sample, method + ".npz")
            code = schema["pore"] if method == "UNet" else 255
            check = polarity(file, human, code)
            m, g = read_mask(file, code)
            if geometry is not None and geometry != g:
                raise ValueError("Unregistered mask grids")
            geometry = g
            volumes[method], data = segment(m)
            del m
            methods[method] = dict(
                source=source, pore_code=code, primary_plane_check=check, **data
            )
        pairs = {
            a + "_vs_" + b: compare(
                volumes[a], volumes[b], methods[a]["eligible"], methods[b]["eligible"]
            )
            for a, b in combinations(METHODS, 2)
        }
        out["specimens"][sample] = dict(
            geometry=geometry, primary_source=hsrc, methods=methods, pairs=pairs
        )
        print(sample, {k: len(v["matches"]) for k, v in pairs.items()}, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    result = run(args.cache)
    with args.out.open("x", encoding="utf8", newline="\n") as fp:
        json.dump(result, fp, separators=(",", ":"), allow_nan=False)
        fp.write("\n")
