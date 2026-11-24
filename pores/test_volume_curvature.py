import numpy as np
import pytest
from pore_curvature_fields import DIR, cpu, field, gpu, scalar


@pytest.mark.parametrize("kind", ["flat", "linear", "sphere", "sheet", "rails"])
def test_curvature_chords_invariance_and_registered_order(kind):
    z, y, x = np.indices((15, 17, 19))
    gen = np.random.default_rng(615)
    raw = {
        "flat": 20000 + z * 0,
        "linear": 1200 + 7 * z + 5 * y + 2 * x,
        "sphere": 1200 + z * z + y * y + x * x,
        "sheet": 1200 + z * z,
        "rails": gen.choice([0, 65535], size=z.shape),
    }[kind].astype("u2")
    f = gpu(raw)
    assert len(DIR) == 13 and np.array_equal(f, cpu(raw))
    for loc in ((0, 0, 0), (2, 3, 4), (4, 6, 8)):
        assert np.array_equal(f[loc], scalar(raw, loc))
    assert np.array_equal(gpu(raw[::-1, :, ::-1]), f[::-1, :, ::-1])
    assert np.array_equal(gpu(raw.transpose(2, 1, 0)), f.transpose(2, 1, 0, 3))
    assert np.array_equal(f, np.concatenate((gpu(raw[:12]), gpu(raw[2:]))))
    if kind in ("flat", "linear"):
        assert not np.any(f)
    if kind == "sphere":
        assert np.all(f == np.repeat([2 * 27 * 6 * r * r for r in (1, 2, 4)], 5))
    if kind == "sheet":
        assert not np.any(f[..., ::5]) and np.all(
            f[..., 4::5] == [2 * 27 * 6 * r * r for r in (1, 2, 4)]
        )


@pytest.mark.parametrize(
    "i,a,b", [(4, 160, 164), (6, 256, 260), (2, 608, 612), (3, 183, 185), (5, 400, 404)]
)
def test_curvature_rejects_unregistered_before_source_reads(i, a, b):
    with pytest.raises(ValueError, match="Unregistered"):
        field(None, i, a, b)


@pytest.mark.parametrize("i,gain", [(2, 87), (3, 139), (5, 15)])
def test_curvature_sealed_masks_and_paired_native_evidence(i, gain):
    from pore_axial_raw import digest, sha
    from pore_axial_score import read_mask
    from pore_region import read
    from pore_volume_curvature import BASE, E
    from test_region_evidence import native_rows

    replay = read(E / "replay.json")
    for k, n in (("raw_retrain_repredict_identical", 10), ("byte_identical_tables", 12)):
        assert len(replay[k]) == n and all(sha(E / f) == h for f, h in replay[k].items())
    src = read(E / "predictions.json")["specimens"][str(i)]
    parent = read(BASE / "predictions.json")["specimens"][str(i)]
    assert all(src[k] == parent[k] for k in ("raw", "segmented", "features_f32"))
    assert src["integer_features_checked"] == 94371840
    d = read(E / f"sample{i}.json")
    hits = {}
    for k, a in d["arms"].items():
        m = src["masks"][k]
        assert m["cpu_gpu_differences"] == m["max_score_difference"] == 0
        assert digest(read_mask(E, i, k, m)) == a["mask_u8_sha256"]
        hits[k] = set(native_rows(E, i, k, a))
    assert hits["curvature"] == hits["profile"] | {gain} and gain not in hits["profile"]
    o = d["arms"]["curvature"]["objects"]
    assert min(o["matched_fraction_prediction"], o["matched_fraction_reference"]) >= 0.9
    assert (
        o["p95_abs_volume_error_pct"]
        > d["arms"]["profile"]["objects"]["p95_abs_volume_error_pct"]
    )
    c = d["comparisons"]["profile"]
    assert c["shared_p95"][1] > c["shared_p95"][0] and not c["advance"]
