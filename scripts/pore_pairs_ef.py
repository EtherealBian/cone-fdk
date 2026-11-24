"""Rebuild two-stage common-reference comparisons. Never combine specimens."""

import argparse
from pathlib import Path

from pore_pairs_de import load, output, paired, summarize


def run(old, new, check=False, *, stages=("e", "f")):
    if tuple(stages) not in (
        ("e", "f"),
        ("e", "g"),
        ("f", "g"),
        ("e", "h"),
        ("f", "h"),
        ("g", "h"),
    ):
        raise ValueError("Unsupported stage pair")
    lhs, rhs = stages
    for i in (2, 3, 5):
        rows = paired(load(old, i), load(new, i))
        orig = output(rows)
        header, rest = orig.split(b"\r\n", 1)
        header = b",".join(
            (
                lhs.encode() + v[1:]
                if v.startswith(b"d_")
                else rhs.encode() + v[1:]
                if v.startswith(b"e_")
                else v
            )
            for v in header.split(b",")
        )
        data = header + b"\r\n" + rest
        path = new / f"sample{i}-paired-{lhs}-{rhs}.csv"
        if check:
            if path.read_bytes() != data:
                raise ValueError(f"Stale {path}")
        else:
            path.write_bytes(data)
        print(i, *summarize(rows), sep=" | ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stages", nargs=2, default=("e", "f"))
    a = ap.parse_args()
    run(a.old, a.new, a.check, stages=a.stages)
