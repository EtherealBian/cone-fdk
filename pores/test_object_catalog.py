from fractions import Fraction
from itertools import combinations

import pytest
from pore_object_catalog import bound, p95


@pytest.mark.parametrize(
    "errs",
    [
        [None] * 10,
        [Fraction(0)] * 8 + [None] * 2,
        [Fraction(k) for k in range(20)],
        [Fraction(15)] * 9 + [None],
        [Fraction(300001, 20000)] * 9 + [None],
        [Fraction(0), Fraction(0), Fraction(12), Fraction(19)] + [Fraction(4)] * 7,
    ],
)
def test_optimistic_subset_bound_by_explicit_selection(errs):
    b = bound(errs)
    vals = [x for x in errs if x is not None]
    k = (9 * len(errs) + 9) // 10
    # Enumerate every selectable intact reference subset that actually meets coverage.
    eligible = [p95(sub) for sz in range(k, len(vals) + 1) for sub in combinations(vals, sz)]
    best = min(eligible, default=None)
    assert (
        b["needed"] == k
        and b["best_p95"] == (float(best) if best is not None else None)
        and b["possible"] == (best is not None and best <= 15)
    )


@pytest.mark.parametrize("i", (2, 3, 5))
def test_catalog_is_reproducible_from_public_inputs_and_independent_csv(i):
    import json

    from pore_object_catalog import ROOT, process

    stored = json.loads((ROOT / "object-catalog" / f"sample{i}.json").read_bytes())
    actual = json.loads(json.dumps(process(i)))
    assert actual == stored and len(actual["input_arms"]) == 33
    assert actual["unique_masks"] == len(actual["independent_csv_by_unique_mask"])
