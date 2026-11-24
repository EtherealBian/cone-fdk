import csv
import json
from itertools import product
from statistics import median

import numpy as np
import pytest
from pore_axial_raw import sha
from pore_contrast_audit import H, shell, summary


def test_shell_of_concave_object_by_point_centers():
    m = np.zeros((3, 4, 5), bool)
    m[1, 0, :] = True
    m[:, 0, 0] = True
    m[1, :, 4] = True
    xyz = np.argwhere(m) + 4
    got = shell(m)
    for q in product(*(range(k) for k in got.shape)):
        assert got[q] == (4 <= np.min(np.sum((xyz - q) ** 2, axis=1)) <= 16)


def test_auc_ties_undefined_and_small_groups():
    rows = [
        dict(cnr=c, matched=b, id=j, voxels=64)
        for j, (c, b) in enumerate([(1, True), (2, True), (2, False), (1, False)])
    ]
    s = summary(rows)
    assert s["auc"] == 0.5 and not s["supported"] and not s["passes"]
    rows[-1]["cnr"] = None
    s = summary(rows)
    assert s["undefined"] == 1 and s["auc"] is None and not s["supported"]


@pytest.mark.parametrize("i", [2, 3, 5])
def test_contrast_populations_and_matching(i):
    a = json.loads((H.parent / "local-contrast" / f"sample{i}.json").read_bytes())
    h = H / f"sample{i}.json"
    report = json.loads(h.read_bytes())
    m = H / f"sample{i}-matches.csv"
    assert (
        sha(h) == a["h_report"]
        and sha(m) == a["matches_file"]
        and a["source_raw"] == report["seal"]["raw"]
        and a["source_segmented"] == report["seal"]["segmented"]
    )
    with m.open(newline="") as f:
        hits = list(csv.DictReader(f))
    for side, id_col in [("reference", "ref_scipy_label"), ("prediction", "pred_id")]:
        rows = a[side]["rows"]
        s = a[side]["summary"]
        ids = {int(x[id_col]) for x in hits}
        assert len({r["id"] for r in rows}) == len(rows) == report["objects"]["eligible_" + side]
        assert ids == {r["id"] for r in rows if r["matched"]}
        for r in rows:
            assert r["voxels"] >= 64 and r["shell_voxels"] > 0 and r["shell_mad"] > 0
            x, y, z = r["source_xyz"]
            assert r["stable_id"] == 1 + x - 360 + (y - 376) * 256 + (z - 400) * 65536
            assert r["cnr"] == (r["shell_median"] - r["inside_median"]) / (
                1.4826 * r["shell_mad"]
            )
            p = r["largest_overlap"]
            if p is not None:
                assert 0 < p["intersection"] <= min(r["voxels"], p["voxels"]) and p[
                    "eligible"
                ] == (not p["touches_face"] and p["voxels"] >= 64)
        g, b = ([r["cnr"] for r in rows if r["matched"] == k] for k in [True, False])
        pairs = len(g) * len(b)
        assert [len(g), len(b), median(g), median(b)] == [
            s[k] for k in ["matched", "unmatched", "median_matched", "median_unmatched"]
        ]
        assert s["pairs"] == pairs and s["twice_wins"] == sum(
            int(x >= y) + int(x > y) for x in g for y in b
        )
        assert (
            s["auc"] == s["twice_wins"] / (2 * pairs)
            and s["supported"]
            and (s["auc"] >= 0.8) == s["passes"]
        )
        assert [r["id"] for r in rows if not r["matched"] and r["cnr"] >= median(g)] == s[
            "unmatched_at_or_above_matched_median"
        ]
    assert (
        a["reference"]["summary"]["unmatched_at_or_above_matched_median"]
        == {2: [86, 114], 3: [140, 147, 169, 220], 5: []}[i]
    )
