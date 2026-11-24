from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from pore_object_catalog import p95
from pore_region_headroom import optimize


@pytest.mark.parametrize("seed", range(6))
def test_exact_minimum_over_all_group_assignments_without_discarding_worse_hits(seed):
    rng = np.random.default_rng(seed)
    gg = [
        [
            [Fraction(int(v), 3) for v in rng.integers(0, 75, size=rng.integers(0, 7))]
            for _ in range(3)
        ]
        for _ in range(7)
    ]
    nref = sum(max(map(len, g)) for g in gg)
    dp, h = optimize(gg, nref)
    best = {}
    for choice in product(range(3), repeat=len(gg)):
        v = [e for g, a in zip(gg, choice, strict=True) for e in g[a]]
        n = len(v)
        s = p95(v)
        if n not in best or (n and s < best[n]):
            best[n] = s
    assert {n: q for n, (q, _) in dp.items()} == best and len(h) == len(gg)
    for n, (q, ch) in dp.items():
        vals = [v for g, a in zip(gg, ch, strict=True) for v in g[a]]
        assert len(vals) == n and p95(vals) == q


def test_all_empty_and_incompatible_same_region_choices():
    e = Fraction(20)
    z = Fraction(0)
    assert optimize([[[], []]], 2)[0][0][0] is None
    result, _ = optimize([[[e, e], [z]], [[z], [z]]], 3)
    assert result[3][0] == 20 and result[2][0] == 0
