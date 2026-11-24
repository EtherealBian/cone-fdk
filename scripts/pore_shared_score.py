"""Check the H shared-model seal and score only its three development fields."""

import argparse
import json
from pathlib import Path

from pore_shared_predict import NAME, SEAL, model_id
from pore_spatial_score import run


def shared_digest(locked):
    rows = locked["specimens"]
    if set(rows) != {"2", "3", "5"} or len({r["weights"] for r in rows.values()}) != 1:
        raise ValueError("H must use one frozen weight archive for all three development scans")
    return rows["2"]["weights"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("baseline", "cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    lock = json.loads(SEAL.read_text(encoding="utf8"))
    shared_digest(lock)
    if lock["identity"] != model_id():
        raise ValueError("H source or runtime differs from seal")
    run(a.baseline, a.cache, a.out, seal=SEAL, method=NAME)
