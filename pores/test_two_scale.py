import numpy as np
import pytest
from pore_raw_grid import GridGPU, cpu_grid, expand, scalar


@pytest.mark.parametrize("s", [1, 4])
def test_scanner_phase_halo_and_independent_scalar(s):
    rng = np.random.default_rng(493)
    raw = rng.integers(0, 65536, (3, 92, 112), dtype="u2")
    full = expand(cpu_grid(raw, s, 3), s)
    for y, x in ((0, 0), (1, 7), (12, 37), (59, 79)):
        np.testing.assert_allclose(full[:, y, x], scalar(raw, s, 3, y, x), atol=1e-5, rtol=0)
    for y, x in ((0, 0), (4, 12), (28, 48)):
        tile = expand(cpu_grid(raw[:, y : y + 64, x : x + 64], s, 3), s)
        np.testing.assert_array_equal(tile, full[:, y : y + 32, x : x + 32])
    # An arbitrary tile offset in the output plane must not change the sampling lattice.
    g = cpu_grid(raw, s, 3)
    np.testing.assert_array_equal(expand(g[:, 3:51, 1:49], s), full[:, 3:35, 1:33])
    np.testing.assert_array_equal(expand(cpu_grid(raw[::-1], s, 3), s), full)


@pytest.mark.parametrize("s", [1, 4])
def test_two_scale_cuda_box_fields(s):
    from pore_axial_raw import PTX

    if not (PTX / "raw-grid.ptx").exists():
        pytest.skip("Requires compiled CUDA grid")
    raw = np.random.default_rng(91).integers(0, 65536, (3, 52, 76), dtype="u2")
    with GridGPU() as gpu:
        actual = gpu(raw, s, 5)
    np.testing.assert_allclose(actual, cpu_grid(raw, s, 5), rtol=0, atol=1e-6)


def test_sealed_tables_ids_and_screens():
    import csv
    import json

    from pore_axial_score import read_mask
    from pore_two_scale import E, sha

    d = json.loads((E / "predictions.json").read_bytes())
    assert d["fits"]["fine"]["init"] == d["fits"]["coarse"]["init"]
    assert set(d["specimens"]) == set("235")
    for k in ("fine", "coarse"):
        assert (
            sha(E / f"{k}.npz") == d["fits"][k]["weights"]
            and d["fits"][k]["trace"][-1][0] == 1535
        )
    for i in (2, 3, 5):
        s = json.loads((E / f"sample{i}.json").read_bytes())
        r = d["specimens"][str(i)]
        v = {}
        masks = {}
        assert not s["advance"] and s["pipeline_id"] == d["pipeline_id"]
        for k, arm in s["arms"].items():
            f = E / f"sample{i}-{k}-pores.csv"
            assert sha(f) == arm["table_sha256"]
            o = arm["objects"]
            rec = r["masks"][k]
            masks[k] = read_mask(E, i, k, rec)
            assert (
                rec["cpu_gpu_decision_disagreements"] == 0
                and rec["first_difference"] is None
                and rec["mask"] == arm["mask_u8_sha256"]
            )
            with f.open(newline="") as fp:
                objs = list(csv.DictReader(fp))
                assert (
                    len(objs) == arm["exact_components"]
                    and sum(int(q["voxels"]) >= 64 and int(q["faces"]) == 0 for q in objs)
                    == o["eligible_prediction"]
                )
            with (E / f"sample{i}-{k}-matches.csv").open(newline="") as fp:
                rows = list(csv.DictReader(fp))
                v[k] = {
                    int(q["ref_scipy_label"]): 100
                    * abs(int(q["pred_voxels"]) - int(q["ref_voxels"]))
                    / int(q["ref_voxels"])
                    for q in rows
                }
            assert (
                len(v[k]) == len(rows) == o["matches"]
                and len(arm["references"]) == o["eligible_reference"]
                and arm["slab_depths"] == [8, 37]
            )
            assert (
                o["matches"] / o["eligible_prediction"] == o["matched_fraction_prediction"]
                and o["matches"] / o["eligible_reference"] == o["matched_fraction_reference"]
            )
        for k, screen in s["screens"].items():
            ids = sorted(v[k].keys() & v["coarse"].keys())
            assert ids == sorted(v[k]) == sorted(v["coarse"]) == screen["shared_ids"]
            np.testing.assert_allclose(
                screen["shared_p95"],
                [np.percentile([v[j][u] for u in ids], 95) for j in (k, "coarse")],
                rtol=0,
                atol=1e-12,
            )
            assert {j for j, b in screen["criteria"].items() if not b} == {
                "p95_gain"
            } and not screen["advance"]
        for k in ("fine", "coarse"):
            assert np.any(masks[k] != masks["census"])
            assert all(
                sum(int(row[1]) for row in checks) == 96
                and max(row[0] for row in checks) == 0
                and max(row[2] for row in checks) < 1e-5
                for checks in (r["grid_audit"][k], d["cal_grid_audits"][str(i)][k])
            )
