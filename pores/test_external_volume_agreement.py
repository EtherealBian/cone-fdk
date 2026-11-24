"""Tests for the external digital-reference sensitivity audit, independent of ConeFDK scores."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pore_external_volume_agreement import (
    METHODS,
    SAMPLES,
    bounds,
    compare,
    decode,
    polarity,
    read_mask,
    segment,
)


def pack(path, array):
    np.savez_compressed(path, volume=array, shape=array.shape, spacing=[5.5e-6] * 3)


def test_foreground_decode_is_method_specific_and_rejects_unknowns():
    with pytest.raises(ValueError):
        decode(np.array([0, 2, 255]), 255)
    with pytest.raises(ValueError):
        decode(np.array([0, 255]), 1)
    assert decode(np.array([0, 1, 2, 3]), 1).tolist() == [False, True, False, False]
    assert decode(np.array([0, 255]), 255).tolist() == [False, True]


def test_deposited_axes_and_polarity_check(tmp_path):
    a = np.zeros((6, 9, 11), "u1")
    a[:, 2:4, 3:7] = 255
    p = tmp_path / "vol.npz"
    pack(p, a)
    roi = [[1, 5], [2, 6], [3, 8]]
    mask, g = read_mask(p, 255, roi)
    assert np.array_equal(mask, a[1:5, 2:6, 3:8] == 255) and g["crop_zyx"] == roi
    human = tmp_path / "human.npz"
    z = np.array([4, 0, 5, 1])

    def labels(invert):
        np.savez(
            human,
            z=z,
            slices=np.where(a[z] == 255, 2 if invert else 1, 1 if invert else 2),
            pore=1,
            metal=2,
            background=3,
        )

    labels(False)
    assert polarity(p, human, 255) == [[h, 8, 91, 8, 0] for h in sorted(z.tolist())]
    labels(True)
    with pytest.raises(ValueError, match="enriched"):
        polarity(p, human, 255)
    assert bounds([257, 99, 101]) == [[0, 256], [25, 74], [25, 75]]


def test_interior_counts_differences_and_interval_intersection():
    a = np.zeros((17, 17, 24), bool)
    a[2:6, 2:6, 2:6] = True
    a[6:10, 6:10, 14:18] = True
    a[:4, 12:16, 4:8] = True
    b = a.copy()
    b[6:10, 6:10, 18:20] = True
    b[11:15, 2:6, 2:6] = True
    la, da = segment(a)
    lb, db = segment(b)
    d = compare(la, lb, da["eligible"], db["eligible"])
    assert da["total_components"] == 3 and len(da["eligible"]) == 2 and len(db["eligible"]) == 3
    assert len(d["matches"]) == 2 and len(d["unmatched_b"]) == 1 and not d["unmatched_a"]
    assert d["b_larger"] == 1 and d["b_smaller"] == 0
    assert (
        d["p95_abs_difference_pct_wrt_a"] == 47.5
        and d["no_common_estimate_within_15pct_of_both"] == 1
    )
    assert (
        compare(np.zeros_like(la), lb, {}, db["eligible"])["p95_abs_difference_pct_wrt_a"] is None
    )


def test_deposited_tables_reconstruct_all_denominators_and_paired_statistics():
    p = (
        Path(__file__).resolve().parents[1]
        / "pores/evidence/external-reference/volume-agreement.json"
    )
    d = json.loads(p.read_bytes())
    assert tuple(d["specimens"]) == SAMPLES
    for s in d["specimens"].values():
        assert set(s["methods"]) == set(METHODS) and len(s["pairs"]) == 3
        for name, v in s["pairs"].items():
            a, b = name.split("_vs_")
            A = s["methods"][a]["eligible"]
            B = s["methods"][b]["eligible"]
            pairs = v["matches"]
            ai = [i for i, _, _ in pairs]
            bj = [j for _, j, _ in pairs]
            assert sorted(ai + v["unmatched_a"]) == sorted(map(int, A)) and sorted(
                bj + v["unmatched_b"]
            ) == sorted(map(int, B))
            vols = [(A[str(i)][0], B[str(j)][0], n) for i, j, n in pairs]
            assert all(0 < n <= min(x, y) and 2 * n >= x + y - n for x, y, n in vols)
            assert (
                sum(max(x, y) * 85 > min(x, y) * 115 for x, y, _ in vols)
                == v["no_common_estimate_within_15pct_of_both"]
            )
            assert (sum(y > x for x, y, _ in vols), sum(y < x for x, y, _ in vols)) == (
                v["b_larger"],
                v["b_smaller"],
            )
            diffs = sorted(100 * abs(y - x) / x for x, y, _ in vols)
            ix, r = divmod(95 * (len(diffs) - 1), 100)
            ref = diffs[ix] + r / 100 * (diffs[min(ix + 1, len(diffs) - 1)] - diffs[ix])
            assert ref == pytest.approx(v["p95_abs_difference_pct_wrt_a"])


def test_native_replay_matches_all_deposited_masks_and_pair_counts():
    folder = Path(__file__).resolve().parent / "evidence/external-reference"
    raw = (folder / "volume-agreement.json").read_bytes()
    specimens = json.loads(raw)["specimens"]
    replay = json.loads((folder / "native-replay.json").read_bytes())
    assert replay["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert replay["specimens"].keys() == specimens.keys()
    for sample, independent in replay["specimens"].items():
        source = specimens[sample]
        assert set(independent["methods"]) == set(METHODS)
        for method, check in independent["methods"].items():
            data = source["methods"][method]
            assert check["slab_depths"] == [5, 31]
            table = folder / "native-csv" / f"{sample}-{method}.csv"
            assert hashlib.sha256(table.read_bytes()).hexdigest() == check["native_csv_sha256"]
            with table.open(newline="") as fp:
                rows = list(csv.DictReader(fp))
            assert len(rows) == check["all_components"] == data["total_components"]
            assert (
                sum(int(r["voxels"]) for r in rows) == check["pore_voxels"] == data["pore_voxels"]
            )
            eligible = sorted(
                [int(r["voxels"]), [int(r[f"seed_{k}"]) for k in "zyx"]]
                for r in rows
                if int(r["voxels"]) >= 64 and not int(r["faces"])
            )
            assert len(eligible) == check["eligible"] and eligible == sorted(
                data["eligible"].values()
            )
        assert independent["replayed_pairs"] == {
            k: len(v["matches"]) for k, v in source["pairs"].items()
        }
