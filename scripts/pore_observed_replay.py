import argparse
from pathlib import Path

from pore_axial_raw import DEV, sha, write


def compare(a, b):
    names = {"support.json", *(f"sample{i}-{k}.csv" for i in DEV for k in ("oof", "reference"))}
    if a.resolve() == b.resolve() or any({f.name for f in p.iterdir()} != names for p in (a, b)):
        raise ValueError("Need two complete and separate audits")
    for n in names:
        if (a / n).read_bytes() != (b / n).read_bytes():
            raise ValueError(f"Different records: {n}")
    return {n: sha(a / n) for n in sorted(names)}


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Verify complete calibration support replays")
    for k in ("first", "second", "out"):
        a.add_argument("--" + k, type=Path, required=True)
    a = a.parse_args()
    write(a.out, compare(a.first, a.second))
