"""Raw-only deployment of the registered one-step effective occupancy operator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, ROOT, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_forward import CONTROL, D, E, calibration, identity, kernel, native
from pore_transfer import FACE
from scipy import ndimage as ndi

ARMS = ("center", "blur")


def id_inverse():
    return dict(
        forward=identity(),
        fit=sha(E / "fit.json"),
        check=sha(E / "check.json"),
        parent=sha(EVID / "predictions.json"),
        native={k: sha(ROOT / "build/pores" / k) for k in ("cone-pores.exe", "pores.ptx")},
        files={
            n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in (
                "scripts/pore_forward_inverse.py",
                "scripts/pore_forward_score.py",
                "scripts/pore_transfer.py",
                "pores/pore_check.py",
            )
        },
    )


def cpu_step(m, raw, q, phases):
    p2, m2 = phases
    contrast = m2 - p2
    w, g = kernel(q)
    norm = int((g * g).sum())
    mi = m.astype("i4")
    # Each convolution sum is an integer <2**53 so scipy's exact-valued double accumulation suffices.
    u = m2 - 2 * raw.astype("i4")
    field = ndi.correlate(mi, g, mode="nearest")
    back = ndi.correlate(field, g, mode="nearest", output=np.int64)
    uns = ndi.correlate(u, g, mode="nearest", output=np.int64)
    faces = ndi.correlate(mi, FACE.astype("i4"), mode="nearest") - mi
    votes = 12 * (D * uns - contrast * back) + contrast * norm * (
        9 * (2 * mi.astype("i8") - 1) + faces - 3
    )
    sl = np.s_[4:-4, 4:-4, 4:-4]
    out = m.copy()
    v = votes[sl]
    out[sl] = (v > 0) | ((v == 0) & m[sl])
    return out, int((v == 0).sum())


def predict(cache, out):
    fit = json.loads((E / "fit.json").read_bytes())
    ck = json.loads((E / "check.json").read_bytes())
    ident = id_inverse()
    assert (
        fit["identity"] == identity()
        and fit["pipeline_id"] == token(identity())
        and ck["fit_sha256"] == sha(E / "fit.json")
        and ck["passed"]
    )
    parent = json.loads((EVID / "predictions.json").read_bytes())["specimens"]
    out.mkdir(parents=True, exist_ok=False)
    spec = {}
    for i in DEV:
        pr = parent[str(i)]
        base = read_mask(EVID, i, "census", pr["masks"]["census"])
        rp, rs = verified(cache, i, "raw", False)
        assert rs["sha256"] == pr["raw"]
        raw = stack(rp, i, "raw", *Z)
        cal = calibration(raw, base)
        rec = dict(
            raw=rs["sha256"],
            segmented=pr["segmented"],
            source_crop=digest(raw),
            census=digest(base),
            predicted_phases=cal,
            masks={},
        )
        for k, q in zip(ARMS, (CONTROL, fit["quantized"]), strict=True):
            ties = None
            if cal["valid"]:
                phases = [cal[name]["median_x2"] for name in ("pore", "material")]
                w, g = kernel(q)
                gpu = native(base, w, raw, phases)
                cpu, ties = cpu_step(base, raw, q, phases)
            else:
                gpu = base
                cpu = base
            diff = int(np.count_nonzero(cpu != gpu))
            assert not diff
            f = out / f"sample{i}-{k}.npz"
            with f.open("xb") as fp:
                np.savez_compressed(fp, mask=gpu)
            rec["masks"][k] = dict(
                file=sha(f),
                mask=digest(gpu),
                cpu_gpu_differences=diff,
                exact_interior_ties=ties,
                additions=int(np.count_nonzero(gpu & ~base)),
                deletions=int(np.count_nonzero(base & ~gpu)),
            )
        spec[str(i)] = rec
        print(i, rec["masks"], flush=True)
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            origin_xyz=[360, 376, 256],
            shape_zyx=list(SHAPE),
            specimens=spec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Do not run with -O")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    predict(a.cache, a.out)
