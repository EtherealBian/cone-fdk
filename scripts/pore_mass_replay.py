"""Independent CPU regeneration: phase erosion and ball dilations, exact signed sums, BFS connectivity."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, ROOT, Z, digest, sha, stack, verified, write
from pore_axial_score import read_mask
from scipy import ndimage as ndi

FACE = ndi.generate_binary_structure(3, 1)
E = ROOT / "pores/evidence"


def ball(r):
    z, y, x = np.ogrid[-r : r + 1, -r : r + 1, -r : r + 1]
    return x * x + y * y + z * z <= r * r


def audit(cache, out):
    seal = json.loads((E / "mass/predictions.json").read_bytes())
    data = {}

    def parent(name, i, k):
        folder = E / name
        spec = json.loads((folder / "predictions.json").read_bytes())["specimens"][str(i)]
        return read_mask(folder, i, k, spec["masks"][k])

    for i in DEV:
        rec = seal["specimens"][str(i)]
        raw, rs = verified(cache, i, "raw", False)
        seg, ss = verified(cache, i, "segmented", False)
        assert (rs["sha256"], ss["sha256"]) == (rec["raw"], rec["segmented"])
        calraw = stack(raw, i, "raw", 160, 184)
        truth = stack(seg, i, "segmented", 160, 184)
        medians = []
        counts = []
        for key, mask in [("pore", truth), ("material", ~truth)]:
            values = np.sort(calraw[ndi.binary_erosion(mask, ball(3))].astype("i8"))
            n = len(values)
            counts.append(n)
            v = int(values[n // 2] + values[(n - 1) // 2]) if n else None
            medians.append(v)
            assert (n, v) == (rec["calibration"][key]["n"], rec["calibration"][key]["median_x2"])
        valid = min(counts) >= 64 and medians[1] > medians[0]
        assert valid == rec["calibration"]["valid"]
        pix = stack(raw, i, "raw", *Z)
        base = parent("census-train", i, "census")
        gray = parent("small-seed", i, "gray")
        assert [digest(v) for v in (calraw, truth, pix, base, gray)] == [
            rec["calibration"]["raw_digest"],
            rec["calibration"]["label_digest"],
            rec["raw_roi"],
            rec["census"],
            rec["gray"],
        ]
        labels, n = ndi.label(base, FACE)
        gp, _ = ndi.label(gray, FACE)
        gboxes = ndi.find_objects(gp)
        gs = np.bincount(gp.ravel())
        archp = E / "mass" / f"sample{i}-estimates.json"
        assert sha(archp) == rec["estimates"]
        archived = {r["seed"]: r for r in json.loads(archp.read_bytes())}
        visited = set()
        entrants = []
        mass = []
        edited = base.copy()
        proposals = {}
        for j, box in enumerate(ndi.find_objects(labels), 1):
            cropped = labels[box] == j
            size = int(cropped.sum())
            origin = np.array([s.start for s in box])
            end = np.array([s.stop for s in box])
            shape = np.array(base.shape)
            if not 8 <= size <= 512 or min(origin) == 0 or min(shape - end) == 0:
                continue
            anchor = tuple(origin + np.argwhere(cropped)[0])
            sid = int(np.ravel_multi_index(anchor, shape)) + 1
            visited.add(sid)
            r = archived[sid]
            gl = gp[anchor]
            old = int(gs[gl])
            assert (r["original"], r["prior_gray"]) == (size, old)
            if size < 64 <= old:
                entrants.append((size, sid))
                gb = gboxes[gl - 1]
                proposals[sid] = (gb, gp[gb] == gl)
            if not valid or min(origin) < 6 or min(shape - end) < 6:
                assert r["estimate"]["reason"] == ("edge" if valid else "calibration")
                continue
            b = tuple(slice(lo - 6, hi + 6) for lo, hi in zip(origin, end, strict=True))
            t = labels[b]
            obj = t == j
            rawbox = pix[b]
            outer = ndi.binary_dilation(obj, ball(6))
            foreign = np.count_nonzero(outer & (t != 0) & ~obj)
            if foreign:
                assert r["estimate"] == dict(reason="neighbor", foreign_sites=foreign)
                continue
            ann = outer & ~ndi.binary_dilation(obj, ball(4))
            inter = ndi.binary_dilation(obj, ball(3))
            annvals = rawbox[ann]
            ivals = rawbox[inter]
            na = len(annvals)
            sa = sum(map(int, annvals))
            num = sum(2 * (sa - int(x) * na) for x in ivals)
            den = 2 * sa - medians[0] * na
            assert (num, den) == (r["estimate"]["numerator"], r["estimate"]["denominator"]) and r[
                "estimate"
            ]["reason"] == ("ok" if den > 0 else "nonpositive_contrast")
            if den <= 0:
                continue
            volume = Fraction(num, den)
            if (size, sid) in entrants and volume >= 64:
                mass.append(sid)
            if size < 64:
                continue
            target = (volume + Fraction(1, 2)).__floor__()
            delta = min(size // 4, max(-(size // 4), target - size))
            assert delta == r["requested_delta"]
            if not delta:
                continue
            shrinking = delta < 0
            fringe = (
                obj & ~ndi.binary_erosion(obj, FACE)
                if shrinking
                else ndi.binary_dilation(obj, FACE) & ~obj
            )
            sites = sorted(
                map(tuple, np.argwhere(fringe)),
                key=lambda q, g=rawbox, sg=(-1 if shrinking else 1): (sg * int(g[q]), q),
            )
            left = abs(delta)
            m = obj.copy()
            for q in sites:
                m[q] = not shrinking
                if shrinking:
                    start = np.zeros_like(m)
                    start.flat[m.argmax()] = True
                    if not np.array_equal(m, ndi.binary_propagation(start, FACE, mask=m)):
                        m[q] = True
                        continue
                left -= 1
                if not left:
                    break
            assert int(m.sum()) - size == r["applied_delta"]
            edited[b] ^= m ^ obj
        assert visited == archived.keys() and n - len(visited) == rec["outside_cohort"]
        size_ids = [sid for _, sid in sorted(entrants, key=lambda q: (-q[0], q[1]))[: len(mass)]]
        masks = {"mass": base.copy(), "size": base.copy(), "boundary": edited}
        for k, v in masks.items():
            for sid in size_ids if k == "size" else mass:
                gb, sel = proposals[sid]
                v[gb] |= sel
        for k, ids in [("mass", mass), ("size", size_ids)]:
            assert set(ids) == {sid for sid, r in archived.items() if r[k]}
        diff = {
            k: int(np.count_nonzero(v != read_mask(E / "mass", i, k, rec["masks"][k])))
            for k, v in masks.items()
        }
        assert not any(diff.values())
        data[str(i)] = dict(
            calibration_counts=counts,
            phase_medians_x2=medians,
            verified_objects=len(visited),
            selected_mass=mass,
            selected_size=size_ids,
            mask_differences=diff,
            replayed_voxels_per_arm=base.size,
        )
        print(i, len(visited), diff, flush=True)
    write(
        out,
        dict(
            pipeline_id=seal["pipeline_id"],
            predictions_sha256=sha(E / "mass/predictions.json"),
            development=data,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Do not run with -O")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    audit(a.cache, a.out)
