import pytest
from pore_pairs_de import paired, summarize


def fixture(matches):
    refs = {3: 80, 4: 90, 5: 100, 6: 120}
    pairs = {
        r: {
            "pred_id": str(r + 10),
            "pred_voxels": str(n),
            "signed_volume_error_pct": str((n / refs[r] - 1) * 100),
        }
        for r, n in matches.items()
    }
    best = {r: (0, 0, 0, "False") for r in refs}
    inter = {(r, r + 10): min(n, refs[r]) for r, n in matches.items()}
    return refs, pairs, best, inter


def test_pairing_uses_reference_and_keeps_every_miss():
    a = fixture({3: 87, 4: 91})
    b = fixture({3: 78, 5: 105})
    rows = paired(a, b)
    assert (
        [r[2] for r in rows] == ["shared", "lost", "gained", "neither"]
        and rows[0][5:8] == [7, 0, 7]
        and rows[0][14:17] == [-2, 2, 0]
    )
    assert summarize(rows) == (
        [1, "8.75%", "2.50%", "1 / 0 / 0", "5", "4", "6"],
        [4, 2, 2, 1, "8.75%", "2.50%"],
    )
    a[0][3] = 99
    with pytest.raises(ValueError, match="sizes differ"):
        paired(a, b)


def test_empty_pair_never_turns_into_zero_error():
    assert summarize(paired(fixture({}), fixture({})))[0][1:3] == ["—", "—"]
