import numpy as np
import pytest
from pore_axial_raw import PTX
from pore_reciprocal import select


@pytest.mark.skipif(
    not all((PTX / f).exists() for f in ("region-medoid.ptx", "cone-pores.exe", "pores.ptx")),
    reason="compile native tools",
)
def test_whole_boundary_handoff_symmetric_preserves_ambiguous_regions(tmp_path):
    a = np.zeros((16, 20, 38), bool)
    b = np.zeros_like(a)
    a[3:7, 3:7, 3:7] = True
    b[3:7, 3:7, 4:8] = True  # two complete 64-voxel objects, IoU = 3/5
    a[8:11, 3:6, 3:6] = b[8:11, 3:6, 3:6] = True  # shared but too small
    a[:4, 3:7, 14:18] = b[:4, 3:7, 14:18] = True  # clipped
    a[6:10, 10:14, 2:6] = True  # one parent only
    a[6:10, 10:14, 16:20] = a[6:10, 10:14, 21:25] = True
    b[6:10, 10:14, 16:25] = True  # 2:1 split, not eligible for exchange
    a[4:8, 4:8, 30:34] = True
    b[4:8, 4:8, 32:36] = True  # complete but IoU = 1/3
    outs, g = select([a, b], (1,) * 3, tmp_path, "test")
    assert g["reasons"] == {
        "replace": 1,
        "clipped": 1,
        "below 64": 1,
        "component count": 2,
        "below IoU": 1,
    }
    delta = np.zeros_like(a)
    delta[3:7, 3:7, [3, 7]] = True
    assert np.array_equal(outs[0], a ^ delta) and np.array_equal(outs[1], b ^ delta)
    assert sum(row["changed_voxels"] for row in g["regions"]) == 32


@pytest.mark.parametrize("i,swaps,delta", [(2, 32, 2130), (3, 15, 478), (5, 8, 271)])
def test_sealed_reciprocal_evidence(i, swaps, delta):
    from pore_axial_raw import digest, sha
    from pore_axial_score import read_mask
    from pore_reciprocal import ARMS, E, parents
    from pore_region import read
    from test_region_evidence import native_rows

    s = read(E / "predictions.json")
    r = s["specimens"][str(i)]
    p = E / f"sample{i}-choices.json"
    assert sha(p) == r["ledger"]
    g = read(p)
    assert [sha(E / f"sample{i}-{k}-native.csv") for k in ("profile", "census", "union")] == g[
        "tables"
    ]
    src, _, _ = parents(i)
    outs = [read_mask(E, i, k, r["masks"][k]) for k in ARMS]
    assert (
        g["reasons"]["replace"] == swaps
        and np.array_equal(src[0] ^ src[1], outs[0] ^ outs[1])
        and np.array_equal(src[0] & src[1], outs[0] & outs[1])
    )
    assert sum(v["changed_voxels"] for v in g["regions"]) == delta
    for v in g["regions"]:
        assert sum(map(sum, v["sizes"])) - v["overlap"] == v["area"]
        if v["reason"] == "replace":
            assert (
                all(len(ss) == 1 and ss[0] >= 64 for ss in v["sizes"])
                and 2 * v["overlap"] >= v["area"]
                and v["changed_voxels"] == v["area"] - v["overlap"]
            )
    d = read(E / f"sample{i}.json")
    assert d["pipeline_id"] == s["pipeline_id"] and d["advance"] == (i == 5)
    for k, m in zip(("profile", "census", *ARMS), (*src, *outs), strict=True):
        arm = d["arms"][k]
        native_rows(E, i, k, arm)
        assert digest(m) == arm["mask_u8_sha256"]
    for j, (k, base) in enumerate(zip(ARMS, ("profile", "census"), strict=True)):
        c = d["comparisons"][k][base]
        assert (
            not c["gained"]
            and not c["lost"]
            and np.count_nonzero(src[j] != outs[j]) == delta
            and d["arms"][k]["objects"]["eligible_prediction"] == g["eligible"][j]
        )
