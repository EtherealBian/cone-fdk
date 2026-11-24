"""Inventory only the deposited human test slices, not method-generated segmentation volumes."""

import argparse
import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np

API = "https://api.figshare.com/v2/articles/31378690/versions/1"
DOI = "10.1184/R1/31378690.v1"
SAMPLES = ("RBF_05", "RBF_09", "RBF_21", "RBF_105")


def load_url(url):
    with urlopen(url, timeout=30) as r:
        return r.read()


def cached(path, url):
    if not path.exists():
        data = load_url(url)
        path.write_bytes(data)
    return path.read_bytes()


def runs(indices):
    groups = []
    for z in sorted(set(indices)):
        if not groups or groups[-1][1] != z:
            groups.append([z, z + 1])
        else:
            groups[-1][1] = z + 1
    return groups


def inventory(cache):
    cache.mkdir(parents=True, exist_ok=True)
    meta = json.loads(cached(cache / "record-v1.json", API))
    if meta["doi"] != DOI:
        raise ValueError("Record version changed")
    pool = [f for f in meta["files"] if f["name"].endswith("_slices.npz")]
    by_sample = defaultdict(list)
    for f in sorted(pool, key=lambda f: f["id"]):
        sample = f["name"].removesuffix("_slices.npz")
        if sample not in SAMPLES:
            raise ValueError("Unexpected specimen")
        data = cached(cache / (str(f["id"]) + ".npz"), f["download_url"])
        if len(data) != f["size"] or hashlib.md5(data).hexdigest() != f["computed_md5"]:
            raise ValueError("Download mismatch")
        with np.load(BytesIO(data), allow_pickle=False) as z:
            masks, indices = z["slices"], z["z"]
            if (
                indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or len(set(indices.tolist())) != len(indices)
                or masks.ndim != 3
                or len(indices) != len(masks)
                or not len(indices)
                or min(indices) < 0
            ):
                raise ValueError("Malformed slice geometry")
            by_sample[sample].append(
                dict(
                    file_id=f["id"],
                    sha256=hashlib.sha256(data).hexdigest(),
                    shape=list(masks.shape),
                    z=indices.tolist(),
                )
            )
    if set(by_sample) != set(SAMPLES) or len(pool) != 12:
        raise ValueError("Incomplete deposit inventory")
    results = {}
    for key in SAMPLES:
        files = by_sample[key]
        if len(files) != 3 or len({tuple(f["shape"][1:]) for f in files}) != 1:
            raise ValueError("Incompatible annotator files")
        contiguous = runs(z for f in files for z in f["z"])
        results[key] = dict(
            test_slice_files=files,
            union_z_runs_half_open=contiguous,
            longest_union_run=max(b - a for a, b in contiguous),
        )
    return dict(
        doi=DOI,
        scope="all deposited human test slice files; union across annotators, not a 3D reference",
        specimens=results,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = inventory(a.cache)
    with a.out.open("x", encoding="utf8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
