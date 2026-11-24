import itertools

import numpy as np
import pytest
from pore_bernsen_provenance import assess, rules


@pytest.mark.parametrize(
    "lo,center,hi,ct,expected",
    [
        (128, 128, 128, 10, (False, True, True)),
        (127, 127, 127, 10, (True, True, True)),
        (125, 129, 129, 5, (True, False, True)),
        (100, 110, 121, 21, (False, True, False)),
        (145, 150, 155, 10, (False, True, False)),
        (145, 151, 155, 10, (False, False, False)),
    ],
)
def test_bernsen_branches_and_unsigned_overflow(lo, center, hi, ct, expected):
    u = np.full((1, 15, 15), center, dtype="u1")
    u[0, 6, 2], u[0, 8, 12] = lo, hi  # (+/-5,+/-1) belongs to ImageJ's radius-5 footprint
    assert tuple(v[0, 7, 7] for v in rules(u, ct)) == expected


def test_transitions_conserve_both_confusion_matrices():
    a, b, t, low = np.array(list(itertools.product((False, True), repeat=4))).T
    r = assess(dict(strict=a, printed=b, plugin=~a, low=low), t)
    assert r["implementation_disagreements"] == 16 and not r["advance"]
    for k in ("fp", "fn"):
        for arm, part in [("strict", "fixed"), ("printed", "added")]:
            assert r[arm][k] == sum(
                v["unchanged_" + k] + v[k + "_" + part] for v in r["transitions"].values()
            )
