"""Replay calibration target bytes by enumerating discrete shells, without a distance transform."""

import argparse
import itertools
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, digest, stack, verified, write
from pore_distance import E, load


def explicit(t):
    if t.ndim != 3 or min(t.shape) < 9 or t.dtype != bool:
        raise ValueError("Expected binary volume with a complete interior radius-four ball")
    d = np.zeros(t.shape, "f4")
    core = np.s_[4:-4, 4:-4, 4:-4]
    c = t[core]
    sq = np.zeros(c.shape, "u1")
    order = sorted(
        (sum(i * i for i in off), off)
        for off in itertools.product(range(-4, 5), repeat=3)
        if 0 < sum(i * i for i in off) <= 16
    )
    for rr, off in order:
        window = tuple(slice(4 + v, n - 4 + v) for n, v in zip(t.shape, off, strict=True))
        newly = (sq == 0) & (c != t[window])
        sq[newly] = rr
    v = sq != 0
    d[core][v] = np.where(c[v], 1.0, -1.0) * (np.sqrt(sq[v].astype("f8")) - 0.5)
    return d


def run(cache, out):
    published = load(E / "census.json")
    rec = {}
    for i in DEV:
        p, ss = verified(cache, i, "segmented", False)
        r = published["specimens"][str(i)]
        assert r["segmented"] == ss["sha256"]
        t = stack(p, i, "segmented", 160, 184)
        assert digest(t) == r["cal_mask"]
        d = explicit(t)
        sig = digest(np.sign(d) * np.float32(0.5))
        matches = digest(d) == r["distance_f32"] and sig == r["sign_f32"]
        if not matches:
            raise ValueError("Enumerated distances differ from sealed byte hashes")
        rec[str(i)] = dict(
            cal_mask=digest(t),
            distance_f32=digest(d),
            sign_f32=sig,
            compared=t.size,
            inside_ball=int(np.count_nonzero(d)),
            offsets=256,
        )
    write(
        out,
        dict(
            method="enumerate the 256 nonzero integer offsets in a radius-four ball; ordered squared distances, half-center shift",
            specimens=rec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
