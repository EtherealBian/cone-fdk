"""Independent body inserts, optional carrier-only slip; ray truth is never a locator input."""

import argparse
from pathlib import Path

import numpy as np
from loss_phantom import counting_scan
from remount_phantom import BALLS, project
from scipy.spatial.transform import Rotation

INSERTS = np.array([[7.55, 0, -5.65], [-4.5, 6.2, 5.40], [-3.0, -7.0, -5.65]])


def generate(root):
    root.mkdir(parents=True, exist_ok=True)
    # Rotation about insert A is a single-witness false pass by construction.
    pivot = INSERTS[0]
    turn = Rotation.from_euler("z", 4.0, degrees=True).as_matrix()
    for name, mount, rack, inserts in [
        ("reference", "first", BALLS, INSERTS),
        ("repeat", "second", BALLS, INSERTS),
        ("changed", "second", BALLS, INSERTS),
        ("slide", "second", BALLS + [0, 0, 0.8], INSERTS),
        ("pivot", "second", (BALLS - pivot) @ turn.T + pivot, INSERTS),
        ("missing", "first", BALLS, INSERTS[[0, 2]]),
    ]:
        project(root / name, mount, relief=name == "changed", rack=rack, witnesses=inserts)
        print(name, flush=True)
    for name, seed in [("reference", 99581), ("changed", 39089)]:
        counting_scan(root / name, root / ("noisy-" + name), seed)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    generate(cli.parse_args().root)
