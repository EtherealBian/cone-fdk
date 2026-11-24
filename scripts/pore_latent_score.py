"""Score only candidate G's publicly sealed three development decision-margin fields."""

import argparse
import json
from pathlib import Path

from pore_latent_predict import NAME, SEAL, model_id
from pore_spatial_score import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("baseline", "cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    locked = json.loads(SEAL.read_text(encoding="utf8"))
    if locked["identity"] != model_id():
        raise ValueError("G runtime or source identity differs from the published seal")
    run(a.baseline, a.cache, a.out, seal=SEAL, method=NAME, field_key="field")
