import numpy as np
import pytest
from pore_cross_depth import eligible, median27
from scipy import ndimage as ndi


def test_uint16_order_filter_matches_scipy_including_edges_and_rail_values():
    rng = np.random.default_rng(561)
    x = rng.integers(0, 65536, (7, 13, 9), dtype="u2")
    x[:, :, 0] = 65535
    x[-3:, :, 5:] = 0
    assert np.array_equal(median27(x), ndi.median_filter(x, size=3, mode="nearest")[1:-1])
    assert np.array_equal(median27(x[1:5]), median27(x)[1:3])
    with pytest.raises(ValueError):
        median27(x.astype("i4"))


@pytest.mark.parametrize(
    "i,a,b",
    [
        (4, 608, 609),
        (6, 256, 260),
        (5, 600, 609),
        (2, 700, 705),
        (2, 400, 404),
        (3, 160, 164),
        (3, 259, 261),
    ],
)
def test_only_two_explicit_development_ranges_allowed(i, a, b):
    with pytest.raises(ValueError):
        eligible(i, a, b)
    eligible(2, 256, 260)
    eligible(5, 608, 704)


@pytest.mark.parametrize(
    "i,gates,matched,eligible_reference", [(2, 9, 50, 54), (3, 5, 16, 19), (5, 5, 7, 8)]
)
def test_cross_depth_outputs_replay_and_reference_denominators(
    i, gates, matched, eligible_reference
):
    from pore_axial_raw import sha
    from pore_axial_score import read_mask
    from pore_cross_depth import ARMS, MODEL, XYZ, E
    from pore_region import read
    from test_region_evidence import native_rows

    s = read(E / "predictions.json")
    r = s["specimens"][str(i)]
    d = read(E / f"sample{i}.json")
    replay = read(E / "replay.json")
    assert (
        s["pipeline_id"] == d["pipeline_id"] == replay["pipeline_id"]
        and list(XYZ) == d["origin_xyz"] == s["origin_xyz"]
        and not d["replicated"]
    )
    prev = read(MODEL / f"sample{i}.json")
    assert d["reference_digest"] != prev["reference_digest"]
    assert (
        r["check_fields_count"] == 4 * 10 * 272**2
        and r["check_mask_comparisons"] == 2 * 4 * 256**2
    )
    for k in ARMS:
        v = r["masks"][k]
        read_mask(E, i, k, v)
        assert not v["cpu_gpu_differences"] and not v["max_score_difference"]
        arm = d["arms"][k]
        native_rows(E, i, k, arm)
        for t in ("matches", "pores"):
            n = f"sample{i}-{k}-{t}.csv"
            assert sha(E / n) == replay["byte_identical_tables"][n]
        for metric, ds in d["depth_comparison"][k].items():
            assert (
                ds["new"] == arm["objects"][metric]
                and ds["familiar"] == prev["arms"][k]["objects"][metric]
                and ds["delta"] == ds["new"] - ds["familiar"]
            )
    arm = d["arms"]["profile"]
    assert (
        sum(arm["gates_on_research_slab"].values()) == gates
        and arm["objects"]["matches"] == matched
        and arm["objects"]["eligible_reference"] == eligible_reference
        and d["depth_comparison"]["profile"]["p95_abs_volume_error_pct"]["delta"] > 0
    )
