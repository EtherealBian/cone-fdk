from itertools import product

import numpy as np
import pytest
from pore_axial_raw import PTX
from pore_local_profile import CHANNEL, ORDER, RAD, library_score, profiles
from pore_profile_gpu import gpu_score, pack
from sklearn.ensemble import HistGradientBoostingClassifier


def test_all_offsets_and_rank_features_against_native_neighbors():
    rng = np.random.default_rng(15)
    x = rng.integers(-20, 20, (10, 22, 24)).astype("f4")
    full = profiles(x).reshape(6, 8, 46)
    for z in range(4):
        assert np.array_equal(profiles(x, z), full[z % 2 :: 2, (z // 2) % 2 :: 2].reshape(-1, 46))
    for y, u in product(range(6), range(8)):
        a = 8 + y
        b = 8 + u
        v = list(x[:, a, b])
        for c, r in product(CHANNEL, RAD):
            pts = sorted(x[c, a + j, b + k] for j, k in product((-r, 0, r), repeat=2))
            v.extend(pts[i] - x[c, a, b] for i in ORDER)
        assert np.array_equal(v, full[y, u])


@pytest.mark.skipif(not (PTX / "local-profile.ptx").exists(), reason="compile local-profile PTX")
def test_tree_export_fp64_exact_at_numeric_branch_and_guard_cycles():
    x = np.arange(-50, 50, dtype="f4").reshape(-1, 1)
    h = HistGradientBoostingClassifier(
        max_iter=3, max_bins=25, min_samples_leaf=3, early_stopping=False, random_state=231
    ).fit(x, x.ravel() > 0)
    nodes = [e[0].nodes for e in h._predictors]
    split = np.float32(nodes[0][0]["num_threshold"])
    probe = np.array(
        [
            [split],
            [np.nextafter(split, np.float32(-np.inf))],
            [np.nextafter(split, np.float32(np.inf))],
        ],
        "f4",
    )
    inter = h._baseline_prediction.item()
    v = h.decision_function(probe)
    assert np.array_equal(v, library_score(probe, nodes, inter)) and np.array_equal(
        v, gpu_score(probe, pack(nodes, 1), inter)
    )
    bad = nodes[0].copy()
    bad[0]["left"] = 0
    with pytest.raises(AssertionError):
        pack([bad], 1)
