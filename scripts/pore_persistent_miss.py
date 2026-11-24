"""Enumerate all source overlaps at catalog-wide persistent misses, including ineligible fragments."""

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, Z, digest, sha, stack, verified, write
from pore_region import EV, PANEL, donor, read
from pore_transfer import FACE, label
from scipy import ndimage as ndi


def overlaps(target, labeled):
    lab, sz, whole, boxes = labeled
    n = int(target.sum())
    if not n:
        raise ValueError("Empty target")
    js, hits = np.unique(lab[target], return_counts=True)
    rows = []
    for j, h in zip(js, hits, strict=True):
        if not j:
            continue
        box = boxes[j - 1]
        loc = lab[box] == j
        pt = tuple(
            k.start + int(v)
            for k, v in zip(box, np.unravel_index(loc.argmax(), loc.shape), strict=True)
        )
        seed = int(np.ravel_multi_index(pt, lab.shape)) + 1
        v = int(sz[j])
        h = int(h)
        q = Fraction(h, v + n - h)
        assert int(np.count_nonzero(loc & target[box])) == h and int(loc.sum()) == v
        rows.append(
            dict(
                seed=seed,
                voxels=v,
                overlap=h,
                clipped=not bool(whole[j]),
                eligible=bool(whole[j] and v >= 64),
                iou=str(q),
                coverage=str(Fraction(h, n)),
                purity=str(Fraction(h, v)),
            )
        )
    rows.sort(key=lambda r: (-Fraction(r["iou"]), r["seed"]))
    best = rows[0] if rows else None
    reason = (
        "no overlap"
        if not best
        else "clipped"
        if best["clipped"]
        else "below 64"
        if best["voxels"] < 64
        else "below IoU"
        if Fraction(best["iou"]) < Fraction(1, 2)
        else "assignment"
    )
    return dict(
        overlap_total=sum(r["overlap"] for r in rows),
        best_seed=best["seed"] if best else None,
        best_reason=reason,
        components=rows,
    )


def stats(a):
    if not len(a):
        return None
    return dict(
        count=len(a),
        minimum=int(a.min()),
        median=float(np.median(a)),
        mean=float(a.mean()),
        maximum=int(a.max()),
    )


def run(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for i in DEV:
        cf = EV / "object-catalog" / f"sample{i}.json"
        cat = read(cf)
        files = {}
        volumes = {}
        for k in ("raw", "segmented"):
            p, rec = verified(cache, i, k, False)
            volumes[k] = stack(p, i, k, *Z)
            files[k] = rec["sha256"]
        t = volumes["segmented"]
        raw = volumes["raw"]
        lab, sizes, whole, boxes = label(t)
        assert digest(t) == cat["reference_digest"]
        ids = sorted(int(j) for j, r in cat["objects"].items() if not r["observations"])
        assert ids == cat["never_matched"]
        cases = {}
        for j in ids:
            assert whole[j] and sizes[j] == cat["objects"][str(j)]["reference_voxels"] >= 64
            b = boxes[j - 1]
            o = (360, 376, Z[0])
            xyz = [[x.start + d, x.stop + d] for x, d in zip(reversed(b), o, strict=True)]
            target = lab == j
            shell = ndi.binary_dilation(target, structure=FACE) & ~t
            cases[j] = dict(
                reference_voxels=int(sizes[j]),
                bounds_xyz=xyz,
                inside=stats(raw[target]),
                material_shell=stats(raw[shell]),
                donors={},
            )
        for study, a in PANEL:
            m, rs, ss = donor(i, study, a)
            assert rs == files["raw"] and ss == files["segmented"]
            lm = label(m)
            hist = read(EV / study / f"sample{i}.json")["arms"][a]["references"]
            for j in ids:
                assert hist[str(j)]["pred"] is None
                cases[j]["donors"][study + "/" + a] = overlaps(lab == j, lm)
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                sources=files,
                catalog_sha256=sha(cf),
                reference_digest=digest(t),
                cases=cases,
            ),
        )
        summary[i] = ids
        for j, c in cases.items():
            print(
                i,
                j,
                c["reference_voxels"],
                c["bounds_xyz"],
                [
                    (
                        h["best_reason"],
                        [(r["voxels"], r["overlap"], r["iou"]) for r in h["components"][:1]],
                    )
                    for h in c["donors"].values()
                ],
                flush=True,
            )
    write(out / "summary.json", dict(diagnostic_only=True, cases=summary))


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.cache, a.out)
