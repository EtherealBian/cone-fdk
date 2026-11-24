import itertools

import numpy as np
from pore_split import split, vertex_cut
from pore_split_score import ledger
from scipy import ndimage as ndi


def dumbbell():
    p = np.zeros((14, 14, 22), "f4")
    p[3:7, 3:7, 2:6] = 0.95
    p[3:7, 3:7, 8:12] = 0.97
    p[5, 5, 6:8] = [0.68, 0.53]
    return p


def test_weak_bridge_and_identity_elsewhere():
    p = dumbbell()
    a, r = split(p)
    ref = p >= 0.90
    j = r[0]
    assert j["decision"] == "edited" and j["proposed_voxels"] == 1 and j["cut_cost"] == 3
    assert np.array_equal(
        np.flatnonzero((p >= 0.5) ^ a), [np.ravel_multi_index((5, 5, 7), p.shape)]
    )
    d = ledger(p >= 0.5, a, ref, r)
    assert d["counts"]["rescued"] == 1 and not d["counts"]["unsupported"]
    d = ledger(p >= 0.5, a, p >= 0.5, r)
    assert (
        d["counts"]["single_split"] == 1
        and d["counts"]["unsupported"] == 1
        and d["counts"]["rescued"] == 0
    )


def test_clipped_single_three_and_guard():
    p = dumbbell()
    p[5, 5, :2] = 0.96
    assert not split(p)[1]
    p = dumbbell()
    p[3:7, 3:7, 8:12] = 0.88
    assert split(p)[1][0]["decision"] == "not flagged"
    p = dumbbell()
    p[3:7, 3:7, 15:19] = 0.99
    p[5, 5, 12:15] = 0.6
    assert split(p)[1][0]["decision"] == "more than two cores"
    p = dumbbell()
    p[3:7, 3:7, 6:8] = 0.6
    assert split(p)[1][0]["decision"] == "cut exceeds 2%"


def test_flow_matches_exhaustive_weak_separator():
    part = np.ones((1, 2, 4), bool)
    p = np.array([[[0.97, 0.60, 0.54, 0.98], [0.96, 0.53, 0.62, 0.94]]], dtype="f4")
    cores = np.zeros_like(part, "i4")
    cores[:, :, 0] = 1
    cores[:, :, -1] = 2
    cut, cost = vertex_cut(part, p, cores, 1, 2)
    costs = 1 + np.floor(100 * (p.astype("f8") - 0.5))
    best = np.inf
    for use in itertools.product([False, True], repeat=4):
        x = np.zeros_like(cut)
        x[cores == 0] = use
        lab, _ = ndi.label(part & ~x)
        if lab[0, 0, 0] != lab[0, 0, 3]:
            best = min(best, costs[x].sum())
    assert cost == best and costs[cut].sum() == best
