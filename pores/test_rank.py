import numpy as np
import pytest
from denoise import Denoise
from pore_check import ROOT
from pore_grid_pilot import grid
from rank_model import edge_weights, features


def test_rank_kernel_against_fractional_tie_scalar():
    rng = np.random.default_rng(431)
    a = rng.normal(20, 1.7, (3, 9, 17)).astype("f4")
    a[0] = 7.5
    h = 1.083
    with Denoise(ROOT / "build/pores/rankpatch.ptx", ranked=True) as gpu:
        result = gpu(a, h)
    for z, y, x in ((0, 0, 0), (1, 0, 0), (2, 5, 8), (2, 8, 16)):
        im = np.pad(a[z].astype(float), 12, mode="edge")
        y += 12
        x += 12
        target = im[y - 2 : y + 3, x - 2 : x + 3]
        buckets = [[] for _ in range(12)]
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                v, u = y + dy, x + dx
                diff = target - im[v - 2 : v + 3, u - 2 : u + 3]
                q = np.mean(diff**2) / h**2
                index = np.searchsorted(2.0 ** np.arange(-5, 6), q, side="right")
                buckets[index].append([im[v, u], im[v, u] + diff.mean()])
        for j, r in enumerate((8, 32, 96)):
            total = np.zeros(2)
            left = r
            for vals in buckets:
                if vals and left:
                    n = min(left, len(vals))
                    total += n * np.mean(vals, axis=0)
                    left -= n
            np.testing.assert_allclose(
                result[z, [j, j + 3], y - 12, x - 12], total / r, rtol=2e-5, atol=2e-5
            )


def test_weight_only_complete_calibration_boundaries():
    a = np.zeros((24, 32, 32), bool)
    a[8:12, 8:12, 8:12] = True
    a[:4, 18:22, 18:22] = True
    a[15, 20, 8] = True
    w = edge_weights(a)
    assert w[8, 9, 9] == w[6, 9, 9] == 4 and w[5, 9, 9] == 1
    assert w[2, 18, 20] == w[15, 20, 8] == 1 and edge_weights(a & False).min() == 1


def test_rank_fields_seam_invariance():
    rng = np.random.default_rng(959)
    m = rng.integers(21000, 55001, (10, 31, 32), dtype="u2")
    field = rng.uniform(24000, 51000, (10, 12, 31, 32)).astype("f4")
    x, mask = features(field, m, 2, halo=9)
    pieces = [features(field[a:b], m[a:b], 2, halo=9) for a, b in [(0, 7), (3, 10)]]
    np.testing.assert_array_equal(x, np.concatenate([v for v, _ in pieces]))
    np.testing.assert_array_equal(mask, np.concatenate([v for _, v in pieces]))


@pytest.mark.parametrize("stride", [1, 2, 3, 4, 5])
def test_lattice_source_coordinates_and_fractional_edges(stride):
    rng = np.random.default_rng(7)
    p = rng.normal(size=(1, 25, 15, 17)).astype("f4")
    origin = (353, 337)
    py = stride - 1
    px = 0
    v = grid(p, stride, py, px, origin)
    for y, x in ((0, 0), (6, 9), (14, 16)):
        values = [
            p[0, 5 * (y - cy + 2) + x - cx + 2, cy, cx]
            for cy in range(max(0, y - 2), min(15, y + 3))
            for cx in range(max(0, x - 2), min(17, x + 3))
            if (origin[0] + cy) % stride == py and (origin[1] + cx) % stride == px
        ]
        assert v[0, y, x] == pytest.approx(np.mean(values) if values else 0, abs=2e-6)
