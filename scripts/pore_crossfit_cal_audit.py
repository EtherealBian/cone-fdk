"""Read only original calibration slices; quantify how withholding a specimen changed its detector."""

import argparse
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, digest, sha, write
from pore_crossfit_gate import CAL, PREV
from pore_region import read
from pore_section_gate import detector, load_source


def counts(m, t):
    return dict(fp=int((m & ~t).sum()), fn=int((~m & t).sum()))


def audit(cache, pred):
    d = read(pred / "predictions.json")
    assert (
        sha(pred / "fit.json") == d["fit"]
        and sha(PREV / "predictions.json") == d["identity"]["previous"]
    )
    fold = read(pred / "fit.json")["calibration"]
    prior = read(PREV / "fit.json")["calibration"]
    assert sha(PREV / "fit.json") == read(PREV / "predictions.json")["fit"]
    result = {}
    for i in DEV:
        key = str(i)
        path, _ = load_source(cache, i, CAL, "raw")
        _, truth = load_source(cache, i, CAL, "segmented")
        old = detector(path, i)
        a, b = prior[key], fold[key]
        assert digest(old) == a["mask"] and digest(truth) == a["reference"] == b["reference"]
        f = pred / f"sample{i}-oof.npz"
        assert sha(f) == b["file"]
        with np.load(f, allow_pickle=False) as z:
            new = z["mask"]
        assert digest(new) == b["mask"] and counts(new, truth) == {k: b[k] for k in ("fp", "fn")}
        changed = old != new
        result[key] = dict(
            original=counts(old, truth),
            held_out=counts(new, truth),
            changed=int(changed.sum()),
            fixed=int((changed & (new == truth)).sum()),
            broken=int((changed & (old == truth)).sum()),
            reference_voxels=int(truth.sum()),
            total_voxels=truth.size,
        )
    return dict(pipeline_id=d["pipeline_id"], calibration_z=list(CAL), specimens=result)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "pred", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    write(a.out, audit(a.cache, a.pred))
