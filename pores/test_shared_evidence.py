import csv
import json
from pathlib import Path
from statistics import quantiles

import pytest

E = Path(__file__).resolve().parent / "evidence"


def read(path):
    return json.loads((E / path).read_text(encoding="utf8"))


@pytest.mark.parametrize("i", ["2", "3", "5"])
def test_sealed_h_replay_and_individual_denominators(i):
    h = read(f"candidate-h/sample{i}.json")
    g = read(f"candidate-g/sample{i}.json")
    m = read("candidate-h/predictions.json")
    replay = read("candidate-h/raw-weight-replay.json")
    assert set(m["specimens"]) == set(replay["specimens"]) == set("235")
    assert len({v["weights"] for v in m["specimens"].values()}) == 1
    assert m["specimens"][i] == h["seal"] and h["source"] == g["source"]
    assert replay["pipeline_id"] == h["pipeline_id"] == m["pipeline_id"]
    r = replay["specimens"][i]
    assert r["raw"] == h["source"]["raw"]["sha256"] and r["mask"] == h["seal"]["mask"]
    assert r["voxels"] == h["native"]["roi_voxels"] == 8388608
    assert r["decision_disagreements"] == h["seal"]["cpu_gpu_decision_disagreements"] == 0
    summary = read("candidate-h/summary.json")[i]
    assert len(h["final_gates"]) == 11 and summary["gates"] == sum(h["final_gates"].values())
    assert summary["failed"] == [k for k, v in h["final_gates"].items() if not v]
    with (E / f"candidate-h/sample{i}-paired-g-h.csv").open(newline="") as f:
        pairs = list(csv.DictReader(f))
    assert (
        len(pairs)
        == len({r["reference_id"] for r in pairs})
        == h["objects"]["eligible_reference"]
    )
    shared = [r for r in pairs if r["status"] == "shared"]
    for k, mode, rec in [("g", "lost", g), ("h", "gained", h)]:
        assert (
            len(shared) + sum(row["status"] == mode for row in pairs) == rec["objects"]["matches"]
        )
        matched = [row for row in pairs if row["status"] in ("shared", mode)]

        def p95(rows, k=k):
            return quantiles(
                [
                    100 * abs(int(row[f"{k}_signed_delta_vox"])) / int(row["reference_voxels"])
                    for row in rows
                ],
                n=100,
                method="inclusive",
            )[94]

        assert p95(matched) == pytest.approx(rec["objects"]["p95_abs_volume_error_pct"])
        if i == "5":
            assert p95(shared) == pytest.approx(9.39119431165 if k == "g" else 18.9550679206)
    if i == "5":
        lost = {row["reference_id"]: row for row in pairs if row["status"] == "lost"}
        assert set(lost) == {"21", "92", "105"}
        assert [int(lost[n]["h_miss_best_iou_pred_vox"]) for n in ("21", "92", "105")] == [
            49,
            72,
            7,
        ]
        assert float(lost["92"]["h_miss_best_iou"]) == pytest.approx(71 / 143)


def test_slice_noise_control_reproduces_baseline_without_advancing():
    old, new = read("search-support.json"), read("slice-noise-check.json")
    for i in "235":
        assert old[i]["check"]["square"]["30"] == new[i]["fixed"]
        assert new[i]["advance"] is False and new[i]["delta"]["dice"] < 0.005
