"""Replay external digital-mask measurements using the native CUDA counter and an alternate reader."""

import argparse
import csv
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
from pore_check import EXE, native, reference  # noqa: E402
from validate import objects  # noqa: E402


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def replay(evidence, cache, out):
    if not __debug__:
        raise ValueError("Use Python without -O for the independent reference checker")
    out.mkdir(parents=True, exist_ok=False)
    archive = json.loads(evidence.read_bytes())
    names = ("Binary", "Adaptive", "UNet")
    report = dict(
        input_sha256=digest(evidence),
        native_sha256=digest(EXE),
        ptx_sha256=digest(EXE.with_name("pores.ptx")),
        specimens={},
    )
    if set(archive["specimens"]) != {"RBF_05", "RBF_09", "RBF_21", "RBF_105"}:
        raise ValueError("Incomplete external specimen set")
    with TemporaryDirectory(dir=cache) as td:
        scratch = Path(td)
        src = scratch / "mask.f32"
        for sample, rec in archive["specimens"].items():
            result = {}
            labs, rows, masks = {}, {}, {}
            for method in names:
                d = rec["methods"][method]
                spec = d["source"]
                p = cache / f"{spec['file_id']}-{method}.npz"
                assert digest(p) == spec["sha256"]
                with np.load(p, allow_pickle=False) as f:
                    # Independent path: decode the full NPZ member before slicing, not via streamed byte offsets.
                    full = f["volume"]
                    z, y, x = full.shape
                    size = (256, y // 2, x // 2)
                    ranges = [
                        ((n - k) // 2, (n - k) // 2 + k)
                        for n, k in zip(full.shape, size, strict=True)
                    ]
                    assert list(map(list, ranges)) == rec["geometry"]["crop_zyx"]
                    mask = full[tuple(slice(a, b) for a, b in ranges)] == (
                        1 if method == "UNet" else 255
                    )
                    del full
                    spacing = tuple(float(v) * 1e6 for v in f["spacing"])
                    assert np.array_equal(f["spacing"], rec["geometry"]["spacing_as_stored"])
                (~mask).astype("<f4").tofile(src)
                table = out / f"{sample}-{method}.csv"
                other = scratch / "second.csv"
                one = native(src, table, mask.shape, spacing, 0, 6, 5, "none", 32)
                two = native(src, other, mask.shape, spacing, 0, 6, 31, "none", 128)
                if (one["slab_depth"], two["slab_depth"]) != (
                    5,
                    31,
                ) or table.read_bytes() != other.read_bytes():
                    raise ValueError("Native slab replay mismatch")
                lab, rs = reference(mask, 6, spacing, table, one)
                recovered = {}
                for r in rs:
                    if int(r["faces"]) == 0 and int(r["voxels"]) >= 64:
                        seed = [int(r["seed_" + k]) for k in "zyx"]
                        recovered[str(lab[tuple(seed)])] = [int(r["voxels"]), seed]
                if (
                    recovered != d["eligible"]
                    or len(rs) != d["total_components"]
                    or one["pore_voxels"] != d["pore_voxels"]
                ):
                    raise ValueError("Difference from streamed crop counts")
                result[method] = dict(
                    slab_depths=[5, 31],
                    all_components=len(rs),
                    eligible=len(recovered),
                    pore_voxels=one["pore_voxels"],
                    native_csv_sha256=digest(table),
                )
                labs[method], rows[method], masks[method] = lab, rs, mask
            matches = {}
            for a, b in combinations(names, 2):
                key = a + "_vs_" + b
                table = scratch / "pairs.csv"
                report_ab = objects(labs[a], masks[b], rows[a], table)
                s = rec["pairs"][key]
                expected = {(i, j): v for i, j, v in s["matches"]}
                seen = set()
                relative_a, relative_b = [], []
                native_to_scipy = {
                    int(r["id"]): int(labs[a][tuple(int(r["seed_" + k]) for k in "zyx")])
                    for r in rows[a]
                }
                with table.open(newline="") as fp:
                    for row in csv.DictReader(fp):
                        ids = (native_to_scipy[int(row["pred_id"])], int(row["ref_scipy_label"]))
                        n = expected[ids]
                        va = int(row["pred_voxels"])
                        vb = int(row["ref_voxels"])
                        assert np.isclose(
                            float(row["iou"]), n / (va + vb - n), rtol=0, atol=1e-14
                        )
                        assert ids not in seen
                        relative_a.append(100 * abs(va - vb) / va)
                        relative_b.append(100 * abs(va - vb) / vb)
                        seen.add(ids)
                assert seen == set(expected) and len(seen) == report_ab["matches"]
                assert (report_ab["eligible_prediction"], report_ab["eligible_reference"]) == (
                    result[a]["eligible"],
                    result[b]["eligible"],
                )
                assert math.isclose(
                    np.percentile(relative_b, 95),
                    report_ab["p95_abs_volume_error_pct"],
                    rel_tol=0,
                    abs_tol=1e-10,
                )
                assert math.isclose(
                    np.percentile(relative_a, 95),
                    s["p95_abs_difference_pct_wrt_a"],
                    rel_tol=0,
                    abs_tol=1e-10,
                )
                matches[key] = len(seen)
            report["specimens"][sample] = dict(methods=result, replayed_pairs=matches)
            print(sample, [(a, v["all_components"]) for a, v in result.items()], flush=True)
    (out / "replay.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8", newline="\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    replay(a.evidence, a.cache, a.out)
