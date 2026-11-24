"""Exact finite-state bound for a shared quantized volume displacement on fixed reference IDs."""

import hashlib
from fractions import Fraction as F

from pore_chord_fraction import rounded
from pore_object_catalog import p95


def audit(pops):
    if not pops or any(
        not rows or any(min(v, u, r) <= 0 for v, u, r in rows) for rows in pops.values()
    ):
        raise ValueError("Nonempty positive volume triples required")
    cuts = {F(0), F(1)}
    lens = {abs(u - v) for rows in pops.values() for v, u, r in rows}
    for d in lens:
        cuts.update(F(2 * k - 1, 2 * d) for k in range(1, d + 1))
    xs = sorted(cuts)
    sites = sorted(cuts | {(a + b) / 2 for a, b in zip(xs[:-1], xs[1:], strict=True)})
    bests = {}
    win = None
    floor = {}
    trace = hashlib.sha256()
    perms = {k: 0 for k in pops}
    both = 0
    for t in sites:
        at = {}
        for k, rows in pops.items():
            q = p95([F(100 * abs(rounded(v, u, t) - r), r) for v, u, r in rows])
            at[k] = q
            perms[k] += int(q <= 15)
            if k not in bests or q < bests[k][0]:
                bests[k] = (q, t)
        x = max(at.values())
        both += int(x <= 15)
        if win is None or x < win[0]:
            win = (x, t, at)
        trace.update((":".join(map(str, [t, *at.values()])) + "\n").encode())
    for k, rows in pops.items():
        ends = [F(100 * min(abs(v - r), abs(u - r)), r) for v, u, r in rows]
        ideal = [F(100 * max(min(v, u) - r, r - max(v, u), 0), r) for v, u, r in rows]
        floor[k] = dict(
            n=len(rows),
            common_best=str(bests[k][0]),
            witness=str(bests[k][1]),
            per_pore_endpoint=str(p95(ends)),
            per_pore_interval=str(p95(ideal)),
            passing_evaluation_sites=perms[k],
        )
        assert p95(ideal) <= min(p95(ends), bests[k][0])
    return dict(
        breakpoints=len(cuts),
        evaluation_sites=len(sites),
        joint_passing_sites=both,
        minimax=str(win[0]),
        witness=str(win[1]),
        at_witness={k: str(v) for k, v in win[2].items()},
        specimens=floor,
        table_sha256=trace.hexdigest(),
    )
