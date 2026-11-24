from fractions import Fraction as F

import numpy as np
import pytest
from pore_chord_fraction import fraction, project, rounded, sources
from pore_fraction_bound import audit


def test_weighted_calibration_minimizes_specimen_balanced_absolute_relative_loss():
    samples = [[(100, 125, 119), (180, 139, 155), (70, 70, 80)], [(68, 100, 72)], []]

    def loss(a):
        return sum(
            sum(abs(v + (u - v) * a - r) / r for v, u, r in rows) / len(rows)
            for rows in samples
            if rows
        )

    candidates = {
        F(0),
        F(1),
        *(
            min(F(1), max(F(0), F(r - v, u - v)))
            for rows in samples
            for v, u, r in rows
            if v != u
        ),
    }
    assert fraction(samples) == min(candidates, key=lambda a: (loss(a), a))
    assert fraction([[], [(65, 65, 75)]]) == 0
    assert fraction([[(2, 4, 2), (2, 4, 4), (2, 4, 4)]]) == 0


def test_mixed_half_ties_are_real_states_not_only_interval_endpoints():
    pops = {"a": [(64, 65, 65)], "b": [(65, 64, 65)]}
    b = audit(pops)
    assert b["minimax"] == "0" and b["witness"] == "1/2" and b["evaluation_sites"] == 5
    assert (rounded(64, 65, F(1, 2)), rounded(65, 64, F(1, 2))) == (65, 65)


def test_exact_enum_against_dense_independent_integer_raster():
    pops = {"a": [(64, 68, 66), (81, 76, 78)], "b": [(100, 102, 110), (91, 88, 90)]}
    b = audit(pops)
    den = 240

    def at(t):
        scores = []
        for pairs in pops.values():
            e = sorted(
                F(100 * abs((2 * (v * den + t * (u - v)) + den) // (2 * den) - r), r)
                for v, u, r in pairs
            )
            scores.append((e[0] + 19 * e[1]) / 20)
        return max(scores)

    assert F(b["minimax"]) == min(at(t) for t in range(den + 1))


def test_fraction_geometry_reproduces_endpoints_and_clipped_components():
    a = np.zeros((15, 15, 15), bool)
    a[4:8, 4:8, 4:8] = True
    a[0, 0, 0] = True
    b = a.copy()
    b[8, 4, 4:8] = True
    g = {"objects": {"2": {"detector": 64, "actual": 68}}}
    for t in (F(0), F(1, 2), F(1)):
        m, d = project(a, b, g, t)
        assert a.sum() + int(4 * t) == m.sum()
        assert m[0, 0, 0] and np.all(m[a]) and not np.any(m & ~b)
        assert d["2"]["actual"] == rounded(64, 68, t)
        if t in (0, 1):
            assert np.array_equal(m, b if t else a)


@pytest.mark.parametrize("i", [4, 6])
def test_no_reserved_source_paths(i):
    with pytest.raises(ValueError):
        sources(i)


@pytest.mark.parametrize("i,n", [(2, 46), (3, 23), (5, 11)])
def test_sealed_fraction_matched_population(i, n):
    from pore_axial_raw import sha
    from pore_chord_fraction import E
    from pore_region import read
    from test_region_evidence import native_rows

    d = read(E / f"sample{i}.json")
    a, b = [native_rows(E, i, k, v) for k, v in d["arms"].items()]
    triples = d["fixed_reference_triples"]
    rows = read(E / f"sample{i}-choices.json")
    assert a.keys() == b.keys() == set(map(int, triples)) and len(b) == n
    for j, (_, v, r) in a.items():
        u = triples[str(j)][1]
        assert triples[str(j)] == [v, u, r] and b[j][1] == rounded(v, u, F(9, 11))
    assert all(row["target"] == row["actual"] for row in rows.values())
    assert not d["advance"] and not d["comparison"]["gained"] and not d["comparison"]["lost"]
    hashes = read(E / "score-replay.json")["byte_identical_tables"]
    assert len(hashes) == 12 and all(sha(E / k) == v for k, v in hashes.items())


def test_complete_finite_bound_and_raw_calibration():
    from pore_chord_fraction import E
    from pore_region import read

    pop = {
        k: list(read(E / f"sample{k}.json")["fixed_reference_triples"].values()) for k in "235"
    }
    b = audit(pop)
    assert b == read(E / "summary.json")["optimistic_bound"]
    assert b["joint_passing_sites"] == 0 and F(b["minimax"]) > 15
    assert all(F(v["per_pore_endpoint"]) < 15 for v in b["specimens"].values())
    f = read(E / "fit.json")
    samples = [
        [(v["detector"], v["full"], v["reference"]) for v in q["rows"].values()]
        for q in f["specimens"].values()
    ]
    assert list(map(len, samples)) == [13, 1, 0] and fraction(samples) == F(f["fraction"]) == F(
        9, 11
    )
