"""A source-independent CI check of native-coordinate sampling and archived marginals."""

import json

import numpy as np
from pore_printed_target_audit import ROOT, axis_centers, file_sha, fixed_bytes, region_stats


def test_display_centers_and_valid_filter():
    for scale in (0.3, 0.446, 0.5, 0.85):
        for origin in (-2.5, 0.25, 31.5):
            expected = [
                (x, int(np.floor((x - origin) / scale + 0.5)))
                for x in range(-10, 85)
                if 5 <= np.floor((x - origin) / scale + 0.5) < 45
            ]
            assert list(axis_centers(5, 45, origin, scale)) == expected
    v = np.full((3, 33, 37), 35000, "u2")
    v[1, 16, 18] = 65535
    np.testing.assert_array_equal(fixed_bytes(v), np.full((7, 11), 85, "u1"))


def test_figure_native_replay_margins():
    e = ROOT / "figure-binary"
    locked = json.loads((e / "source.json").read_bytes())
    score = json.loads((e / "check.json").read_bytes())
    audit = json.loads((e / "replay.json").read_bytes())
    assert (
        file_sha(e / "native.npz") == locked["file"]
        and file_sha(e / "check.json") == audit["score_sha256"]
        and file_sha(e / "source.json") == audit["source_sha256"]
    )
    with np.load(e / "native.npz", allow_pickle=False) as a:
        for part, row in score["partitions"].items():
            cells = row["archive_fixed_printed"]
            middle = row["intermediate_tones"]
            assert all(0 <= b <= c for b, c in zip(middle, cells, strict=True))
            assert (
                row == region_stats(cells, middle) and row["sufficient"] and not row["concordant"]
            )
            rep = audit["partitions"][part]
            assert (
                rep["archive_bit_differences"] == rep["cpu_filter_bit_differences"] == 0
                and rep["xyuv_sha256"] == locked["arrays"][part + "_xyuv"]
                and all(rep[k] == v for k, v in row.items())
            )
            xy = a[part + "_xyuv"]
            assert (
                len(np.unique(xy[:, :2], axis=0))
                == len(np.unique(xy[:, 2:], axis=0))
                == row["count"]
            )
            assert xy[:, 0].min() >= 616
            labels = a[part + "_archive"].astype(int) * 2 + a[part + "_fixed"]
            np.testing.assert_array_equal(
                np.bincount(labels, minlength=4), np.array(cells).reshape(4, 2).sum(1)
            )
