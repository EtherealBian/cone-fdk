from fractions import Fraction

import pytest
from pore_external_slice_inventory import runs
from pore_retained_baseline import EVID, compare, exact_tail, read


def row(n, p, r):
    return dict(pred_id=n, ref_scipy_label=n, pred_voxels=p, ref_voxels=r, iou=0.5)


def test_tail_uses_fractional_volume_errors_and_linear_interpolation():
    assert exact_tail([row(1, 70, 100), row(2, 100, 100)]) == Fraction(57, 2)
    assert exact_tail([row(1, 64, 80)]) == 20


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [row(1, 64, 64), row(1, 100, 100)],
        [row(1, 63, 64)],
        [dict(row(1, 64, 64), iou=0.4999)],
    ],
)
def test_dont_rank_empty_unmatched_or_ineligible_objects(rows):
    with pytest.raises(ValueError):
        exact_tail(rows)


def test_published_comparison_uses_only_full_development_triples():
    d = compare()
    assert d["minimizers"] == ["E"]
    assert d["candidates"]["E"]["worst_exact_pct"] == "3605/128"
    for v in d["candidates"].values():
        assert set(v["specimens"]) == {"2", "3", "5"}
        assert all(row["failed"] for row in v["specimens"].values())
    archive = read(EVID / "retained-e/summary.json")
    assert {k: v for k, v in archive.items() if k != "source_replay_e"} == d
    for i, v in archive["source_replay_e"]["specimens"].items():
        doc = read(EVID / f"candidate-e/sample{i}.json")
        assert v["review_sha256"] == doc["review_sha256"]
        assert (
            v["matches"] == doc["objects"]["matches"]
            and v["failed"] == d["candidates"]["E"]["specimens"][i]["failed"]
        )


def test_multiple_raters_and_repeated_planes_do_not_inflate_depth():
    assert runs([4, 1, 2, 3, 2, 5, 7, 8, 7]) == [[1, 6], [7, 9]]
    d = read(EVID / "retained-e/external-slices.json")["specimens"]
    assert len(d) == 4
    for v in d.values():
        pool = [z for f in v["test_slice_files"] for z in f["z"]]
        assert len(pool) == 12 and len(set(pool)) == 4
        assert runs(pool) == v["union_z_runs_half_open"] and v["longest_union_run"] == 1
