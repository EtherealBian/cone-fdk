"""Restore integer reference IDs in prior JSON rows before invoking the sealed scorer."""

import argparse
from pathlib import Path

import pore_monotone_score as sealed
from pore_monotone import E, load


def read_typed(folder, name):
    data = load(folder, name)
    if folder == E and name in {f"sample{i}.json" for i in (2, 3, 5)}:
        for arm in data["arms"].values():
            refs = arm["references"]
            typed = {int(k): v for k, v in refs.items()}
            if len(typed) != len(refs) or any(str(int(k)) != k for k in refs):
                raise ValueError("Noncanonical archived reference ID")
            arm["references"] = typed
    return data


def run(cache, out):
    original = sealed.load
    try:
        sealed.load = read_typed
        sealed.run(cache, out)
    finally:
        sealed.load = original


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    run(a.cache, a.out)
