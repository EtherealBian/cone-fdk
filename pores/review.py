"""Self-contained offline review; one contrast-windowed PNG and two exact RLE masks per Z plane."""

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def runs(a):
    flat = a.ravel()
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    end = np.r_[starts[1:], flat.size]
    nz = flat[starts] != 0
    triples = (
        np.stack((starts[nz], end[nz] - starts[nz], flat[starts[nz]]), axis=1)
        .astype("int64")
        .tolist()
    )
    check = np.zeros_like(flat)
    for i, n, v in triples:
        check[i : i + n] = v
    np.testing.assert_array_equal(flat, check)
    return triples


def write_review(path, specimen, raw, ref, lab, rows, origin, spacing):
    lo, hi = np.percentile(raw, [1, 99])
    records = {}
    lookup = np.zeros(int(lab.max()) + 1, dtype=np.int64)
    for r in rows:
        x, y, z = (int(r["seed_" + k]) for k in "xyz")
        tag = int(lab[z, y, x])
        id_ = int(r["id"])
        if tag == 0 or lookup[tag] != 0 or id_ != 1 + np.ravel_multi_index((z, y, x), raw.shape):
            raise ValueError("Bad seed mapping")
        lookup[tag] = id_
        records[id_] = dict(row=r, best=-1, peak=0, first=z, last=z, reviewed_voxels=0)
    layers = []
    for z, (image, labels, truth) in enumerate(zip(raw, lab, ref, strict=True)):
        view = np.clip((image.astype(float) - lo) * 255 / max(hi - lo, 1), 0, 255).astype("u1")
        f = io.BytesIO()
        Image.fromarray(view).save(f, format="PNG")
        pixels = base64.b64encode(f.getvalue()).decode()
        for tag, count in zip(*np.unique(labels[labels != 0], return_counts=True), strict=True):
            o = records[lookup[tag]]
            o["last"] = z
            o["reviewed_voxels"] += int(count)
            if count > o["peak"]:
                o["peak"] = int(count)
                o["best"] = z
        layers.append(dict(image=pixels, objects=runs(lookup[labels]), reference=runs(truth)))
    for o in records.values():
        if o["peak"] <= 0 or o["reviewed_voxels"] != int(o["row"]["voxels"]):
            raise ValueError("Incomplete review coverage")
    payload = dict(
        specimen=specimen,
        origin=origin,
        spacing=spacing,
        low=lo,
        high=hi,
        shape=raw.shape,
        layers=layers,
        objects=records,
    )
    template = (Path(__file__).with_name("review.html")).read_text(encoding="utf8")
    path.write_text(
        template.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))),
        encoding="utf8",
    )
    return dict(
        objects=len(records),
        layers=len(layers),
        mask_encoding="lossless start/length/stable-ID; both volumes round-trip checked",
        display_window_uint16=[lo, hi],
    )
