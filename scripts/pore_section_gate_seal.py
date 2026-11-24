"""Verify that two source replays produced byte-identical preregistered development predictions."""

import argparse
import hashlib
import json
from pathlib import Path
from shutil import copyfile


def seal(first, second, out):
    if first.resolve() == second.resolve() or out.exists():
        raise ValueError("Use two separate source runs and a new output directory")
    names = {
        "fit.json",
        "predictions.json",
        *(f"sample{i}-{suffix}" for i in (2, 3, 5) for suffix in ("choices.json", "gate.npz")),
    }
    if not all({p.name for p in d.iterdir()} == names for d in (first, second)):
        raise ValueError("Incomplete or unexpected prediction set")
    hashes = {}
    for n in sorted(names):
        a = (first / n).read_bytes()
        b = (second / n).read_bytes()
        if a != b:
            raise ValueError(f"Prediction replay differs: {n}")
        hashes[n] = hashlib.sha256(a).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    for n in hashes:
        copyfile(first / n, out / n)
    with (out / "prediction-replay.json").open("x", encoding="utf8", newline="\n") as f:
        json.dump(hashes, f, indent=2)
        f.write("\n")
    return hashes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("first", "second", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    print(len(seal(a.first, a.second, a.out)), "identical source artifacts")
