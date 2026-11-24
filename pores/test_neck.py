import numpy as np
from denoise import Denoise
from neck_model import features
from pore_check import ROOT


def test_adaptive_matches_scalar_patch_reference():
    a = np.random.default_rng(874).normal(size=(3, 7, 17)).astype("f4")
    h = 1.17
    with Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as gpu:
        result = gpu(a, h)
    for z, y, x in [(0, 0, 0), (1, 3, 8), (2, 6, 16)]:
        image = np.pad(a[z].astype("f8"), 12, mode="edge")
        y0, x0 = y + 12, x + 12
        target = image[y0 - 2 : y0 + 3, x0 - 2 : x0 + 3]
        values = []
        weights = []
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                v, u = y0 + dy, x0 + dx
                d = target - image[v - 2 : v + 3, u - 2 : u + 3]
                distance = np.mean(d * d) / h**2
                weights.append(
                    np.exp(
                        -np.array(
                            [
                                2 * distance,
                                distance,
                                4 * max(0, distance - 0.6),
                                6 * max(0, distance - 1.25),
                                2.4 * (abs(d).mean() / h) ** 2,
                                2 * max(0, distance - 0.75 * (d.mean() / h) ** 2)
                                + 0.0125 * (dx * dx + dy * dy),
                            ]
                        )
                    )
                )
                values.append(image[v, u])
        np.testing.assert_allclose(
            result[z, :, y, x],
            np.average(np.tile(values, (6, 1)), axis=1, weights=np.array(weights).T),
            rtol=2e-5,
            atol=2e-5,
        )


def test_axial_features_keep_real_neighbors_across_batch_seam():
    rng = np.random.default_rng(38)
    m = rng.integers(20000, 55001, size=(10, 28, 29), dtype="u2")
    field = rng.uniform(24000, 51000, size=(10, 6, 28, 29)).astype("f4")
    whole, mask = features(field, m, 2, halo=8)
    pieces = [features(field[a:b], m[a:b], 2, halo=8) for a, b in [(0, 7), (3, 10)]]
    np.testing.assert_array_equal(np.concatenate([b for _, b in pieces]), mask)
    np.testing.assert_array_equal(np.concatenate([x for x, _ in pieces]), whole)
