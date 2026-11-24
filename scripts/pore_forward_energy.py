"""Post-score diagnostic: did the simultaneous update lower its own objective? No labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, Z, digest, sha, stack, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_forward import CONTROL, D, E, kernel
from scipy import ndimage as ndi


def energy(mask, prior, u, q):
    _, g = kernel(q)
    s = float(np.sum(g * g)) / (D * D)
    residual = u - ndi.correlate(mask.astype("i4"), g, mode="nearest") / D
    data = float(np.sum(residual * residual, dtype="f8"))
    cuts = int(sum(np.count_nonzero(np.diff(mask, axis=ax)) for ax in range(3)))
    changed = int(np.count_nonzero(mask != prior))
    n = mask.size
    return dict(
        raw_mse=data / n,
        cut_edges=cuts,
        edge_penalty=s * cuts / (12 * n),
        changed=changed,
        prior_penalty=s * changed / (2 * n),
        total=(data + s * cuts / 12 + s * changed / 2) / n,
    )


def diagnose(cache, out):
    seal = json.loads((E / "predictions.json").read_bytes())
    fit = json.loads((E / "fit.json").read_bytes())
    prior = json.loads((EVID / "predictions.json").read_bytes())["specimens"]
    spec = {}
    for i in DEV:
        r = seal["specimens"][str(i)]
        rp, rs = verified(cache, i, "raw", False)
        assert rs["sha256"] == r["raw"]
        raw = stack(rp, i, "raw", *Z)
        assert digest(raw) == r["source_crop"]
        base = read_mask(EVID, i, "census", prior[str(i)]["masks"]["census"])
        assert digest(base) == r["census"]
        p, m = [r["predicted_phases"][k]["median_x2"] for k in ("pore", "material")]
        u = (m - raw.astype("f8") * 2) / (m - p)
        far_bg = ndi.distance_transform_edt(~base) > 4
        far_fg = ndi.distance_transform_edt(base) > 4
        arms = {}
        for k, q in [("center", CONTROL), ("blur", fit["quantized"])]:
            v = read_mask(E, i, k, r["masks"][k])
            a, b = (energy(s, base, u, q) for s in (base, v))
            arms[k] = dict(
                before=a,
                after=b,
                delta=b["total"] - a["total"],
                added_far=int(np.count_nonzero(v & ~base & far_bg)),
                deleted_far=int(np.count_nonzero(~v & base & far_fg)),
            )
        spec[str(i)] = arms
        print(i, arms["blur"], flush=True)
    write(
        out,
        dict(
            predictions_sha256=sha(E / "predictions.json"),
            posthoc=True,
            normalization="per ROI voxel; nearest extension; unordered in-volume face edges; only interior lattice changed",
            specimens=spec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    diagnose(a.cache, a.out)
