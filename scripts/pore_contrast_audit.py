"""Audit every eligible formal H object against native intensities; no fitted decision rule."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, ROOT, digest, sha, stack, verified, write
from pore_transfer import label
from scipy import ndimage as ndi

H = ROOT / "pores/evidence/candidate-h"


def shell(obj):
    """Exact annulus in a true four-voxel surrounding margin."""
    if obj.ndim != 3 or not obj.any():
        raise ValueError("Missing support")
    padded = np.pad(obj, 4)
    dist = ndi.distance_transform_edt(~padded)
    a = (dist >= 2) & (dist <= 4)
    inside = np.zeros_like(a)
    outside = np.zeros_like(a)
    for dz in range(-4, 5):
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                q = dz * dz + dy * dy + dx * dx
                if q > 16:
                    continue
                view = tuple(
                    slice(4 + v, 4 + v + n) for v, n in zip((dz, dy, dx), obj.shape, strict=True)
                )
                outside[view] |= obj
                if q < 4:
                    inside[view] |= obj
    if not np.array_equal(a, outside & ~inside):
        raise ValueError("Integer-offset shell differs")
    return a


def summary(rows):
    groups = [[r["cnr"] for r in rows if r["matched"] == w] for w in (True, False)]
    ok = all(len(g) >= 3 and all(x is not None for x in g) for g in groups)
    med = [float(np.median(g)) if g and all(x is not None for x in g) else None for g in groups]
    pairs = 0
    twice = 0
    if all(x is not None for g in groups for x in g):
        for x in groups[0]:
            for y in groups[1]:
                twice += int(x > y) * 2 + int(x == y)
                pairs += 1
    auc = twice / (2 * pairs) if pairs else None
    return dict(
        matched=len(groups[0]),
        unmatched=len(groups[1]),
        median_matched=med[0],
        median_unmatched=med[1],
        undefined=sum(x is None for g in groups for x in g),
        pairs=pairs,
        twice_wins=twice,
        auc=auc,
        supported=ok,
        passes=bool(ok and auc is not None and auc >= 0.8),
        size_bands={
            s: [
                sum(a <= r["voxels"] < b and r["matched"] == m for r in rows)
                for m in (True, False)
            ]
            for s, a, b in [("64..127", 64, 128), ("128..511", 128, 512), ("512+", 512, 10**10)]
        },
        unmatched_at_or_above_matched_median=[
            r["id"]
            for r in rows
            if not r["matched"]
            and med[0] is not None
            and r["cnr"] is not None
            and r["cnr"] >= med[0]
        ],
    )


def catalog(info):
    lab, sizes, whole, boxes = info
    rec = {}
    for i, box in enumerate(boxes, 1):
        mask = lab[box] == i
        p = np.unravel_index(int(mask.argmax()), mask.shape)
        pos = tuple(int(j + s.start) for j, s in zip(p, box, strict=True))
        rec[i] = (int(np.ravel_multi_index(pos, lab.shape)) + 1, pos)
    return rec


def audit(cache, review_folder, out, i):
    from audit import decode
    from local_model import context_stack

    if i not in DEV:
        raise ValueError("Development only")
    report = json.loads((H / f"sample{i}.json").read_bytes())
    assert report["origin_xyz"] == [360, 376, 400] and report["role"] == "development"
    raw, rs = verified(cache, i, "raw", False)
    seg, ss = verified(cache, i, "segmented", False)
    assert [rs["sha256"], ss["sha256"]] == [report["seal"][k] for k in ("raw", "segmented")]
    rp = review_folder / f"sample{i}-review.html"
    assert sha(rp) == report["review_sha256"]
    payload = json.loads(
        re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            rp.read_text(encoding="utf8"),
            re.S,
        )[1]
    )
    assert payload["origin"] == report["origin_xyz"] and len(payload["layers"]) == 128
    ids = np.stack(
        [decode(q["objects"], 65536).astype("i4").reshape(256, 256) for q in payload["layers"]]
    )
    del payload
    ref = stack(seg, i, "segmented", 400, 528)
    pred = ids != 0
    assert digest(pred) == report["seal"]["mask"] and digest(ref) == report["reference_mask"]
    pred_info = label(pred)
    ref_info = label(ref)
    pix = context_stack(raw, i, 397, 531, 4)
    assert pix.shape == (136, 264, 264)
    f = H / f"sample{i}-matches.csv"
    with f.open(newline="") as fp:
        hits = list(csv.DictReader(fp))
    matched = {
        name: {int(r[key]) for r in hits}
        for name, key in [("reference", "ref_scipy_label"), ("prediction", "pred_id")]
    }
    cats = {
        name: catalog(info) for name, info in [("reference", ref_info), ("prediction", pred_info)]
    }
    res = {}
    for name, info, other, othername in [
        ("reference", ref_info, pred_info, "prediction"),
        ("prediction", pred_info, ref_info, "reference"),
    ]:
        lab, size, whole, boxes = info
        ol, os, ow, _ = other
        rows = []
        for j in np.flatnonzero(whole & (size >= 64)):
            box = boxes[j - 1]
            m = lab[box] == j
            stable, pos = cats[name][j]
            nativeid = stable if name == "prediction" else int(j)
            if name == "prediction":
                assert np.all(ids[box][m] == stable)
            ann = shell(m)
            part = pix[tuple(slice(s.start, s.stop + 8) for s in box)]
            core = part[4:-4, 4:-4, 4:-4][m]
            ring = part[ann]
            a, b = map(float, (np.median(core), np.median(ring)))
            mad = float(np.median(np.abs(ring.astype("f8") - b)))
            ov = np.bincount(ol[box][m], minlength=len(os))
            ov[0] = 0
            jj = int(ov.argmax())
            otherid = (
                (cats[othername][jj][0] if othername == "prediction" else jj) if jj else None
            )
            rows.append(
                dict(
                    id=nativeid,
                    stable_id=stable,
                    source_xyz=[360 + pos[2], 376 + pos[1], 400 + pos[0]],
                    voxels=len(core),
                    shell_voxels=len(ring),
                    inside_median=a,
                    shell_median=b,
                    shell_mad=mad,
                    cnr=(b - a) / (1.4826 * mad) if mad else None,
                    matched=nativeid in matched[name],
                    largest_overlap=None
                    if not jj
                    else dict(
                        id=otherid,
                        intersection=int(ov[jj]),
                        voxels=int(os[jj]),
                        touches_face=not bool(ow[jj]),
                        eligible=bool(ow[jj] and os[jj] >= 64),
                    ),
                )
            )
        r = summary(rows)
        assert (
            len(rows) == report["objects"]["eligible_" + name]
            and r["matched"] == report["objects"]["matches"]
            and matched[name] == {q["id"] for q in rows if q["matched"]}
        )
        res[name] = dict(summary=r, rows=rows)
    write(
        out / f"sample{i}.json",
        dict(
            specimen=i,
            origin_xyz=report["origin_xyz"],
            source_raw=rs["sha256"],
            source_segmented=ss["sha256"],
            h_report=sha(H / f"sample{i}.json"),
            matches_file=sha(f),
            raw_halo_zyx=digest(pix),
            **res,
        ),
    )
    print(i, {k: r["summary"] for k, r in res.items()}, flush=True)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--reviews", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for i in DEV:
        audit(args.cache, args.reviews, args.out, i)
