"""Clean-clone acceptance: reproduce the measured result and refuse tampered evidence."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DATA = Path(__file__).resolve().parent
REVIEW = DATA.parents[1] / "platform/spark/review_grid_oracle.py"
PIN = "445b3a5ad5a6585b9621a7d57ca6fb1661e403b27714613e997907449cf730ac"


def check(pin=PIN, native=DATA / "native.mg", witness=DATA / "astra.mg"):
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(REVIEW),
            str(DATA / "qualification.json"),
            pin,
            str(native),
            str(witness),
        ],
        capture_output=True,
    )


class PublicWitness(unittest.TestCase):
    def test_all_planes(self):
        proc = check()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = json.loads(proc.stdout)["regions"]
        self.assertEqual([r["voxels"] for r in rows], [64077, 74871])
        self.assertEqual([r["verified_planes"] for r in rows], [31, 27])

    def refused(self, proc):
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, b"")

    def test_wrong_pin(self):
        self.refused(check(pin="0" * 64))

    def test_swapped_arrays(self):
        self.refused(check(native=DATA / "astra.mg", witness=DATA / "native.mg"))

    def test_changed_voxel(self):
        with tempfile.TemporaryDirectory() as wd:
            path = Path(wd) / "modified.mg"
            wire = bytearray((DATA / "native.mg").read_bytes())
            wire[-1] ^= 1
            path.write_bytes(wire)
            self.refused(check(native=path))


if __name__ == "__main__":
    unittest.main()
