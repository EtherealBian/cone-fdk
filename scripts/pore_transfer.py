"""Paired, leave-one-scan-out boundary transfer. Prediction functions take no target labels."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pore_transfer_predict import DEV, ROOT, SHAPE, Z
from scipy import ndimage as ndi

sys.path.insert(0, str(ROOT / "pores"))
from pore_check import native, reference  # noqa: E402
from validate import SPEC, final_gates, objects, score, sha, stack, verified  # noqa: E402

FACE = ndi.generate_binary_structure(3, 1)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view("u1")).hexdigest()


def label(mask):
    lab, n = ndi.label(mask, FACE)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    boxes = ndi.find_objects(lab)
    whole = np.array(
        [False]
        + [
            all(t.start > 0 and t.stop < dim for t, dim in zip(box, lab.shape, strict=True))
            for box in boxes
        ]
    )
    return lab, sizes, whole, boxes


def pairs(pred, ref, minimum, maximum=None):
    lp, vp, wp, _ = pred
    lr, vr, wr, _ = ref
    ep = wp & (vp >= minimum)
    if maximum is not None:
        ep &= vp <= maximum
    er = wr & (vr >= minimum)
    mask = ep[lp] & er[lr]
    keys, hits = np.unique(lp[mask].astype("i8") * len(vr) + lr[mask], return_counts=True)
    edges = []
    for key, hit in zip(keys, hits, strict=True):
        p, r = divmod(int(key), len(vr))
        union = int(vp[p] + vr[r] - hit)
        if 2 * hit >= union:
            edges.append((int(hit) / union, p, r))
    used = set()
    matched = {}
    for iou, p, r in sorted(edges, reverse=True):
        if r not in matched and p not in used:
            matched[r] = (p, iou)
            used.add(p)
    return matched


def boundaries(prob, pred):
    lab, sizes, whole, boxes = pred
    result = {}
    for pid in np.flatnonzero(whole & (sizes >= 32) & (sizes <= 256)):
        box = tuple(
            slice(max(0, t.start - 2), min(dim, t.stop + 2))
            for t, dim in zip(boxes[pid - 1], lab.shape, strict=True)
        )
        local = lab[box]
        obj = local == pid
        inner = obj & ~ndi.binary_erosion(obj, structure=FACE)
        foreign = (local != 0) & ~obj
        outer = (
            (local == 0)
            & ndi.binary_dilation(obj, structure=FACE)
            & ~ndi.binary_dilation(foreign, structure=FACE)
        )

        def indices(mask, box=box):
            pos = tuple(axis + t.start for axis, t in zip(np.nonzero(mask), box, strict=True))
            return np.ravel_multi_index(pos, lab.shape)

        ii, oo = indices(inner), indices(outer)
        s = len(ii)
        u = (prob.flat[oo].sum(dtype="f8") - (1.0 - prob.flat[ii].astype("f8")).sum()) / s
        result[int(pid)] = (ii, oo, float(u))
    return result


def checked_features(prob, pred):
    from pore_transfer_gpu import owners

    lab, sizes, whole, _ = pred
    face_map = owners(
        lab, whole & (sizes >= 32) & (sizes <= 256), ROOT / "build/pores/transfer_faces.ptx"
    )
    cpu = boundaries(prob, pred)
    gpu = {}
    indices = np.flatnonzero(face_map)
    at = face_map.ravel()[indices]
    for k, (ii, oo, u) in cpu.items():
        inn, outer = indices[at == -k], indices[at == k]
        if not np.array_equal(inn, ii) or not np.array_equal(outer, oo):
            raise ValueError(f"CPU/GPU faces differ for component {k}")
        gpu[k] = (inn, outer, u)
    if sum(len(ii) + len(oo) for ii, oo, _ in gpu.values()) != len(indices):
        raise ValueError("Unaccounted CUDA shell voxels")
    return gpu


def donor_rows(pred, ref, features):
    rows = []
    for r, (p, iou) in sorted(pairs(pred, ref, 32, 256).items()):
        inner, _, u = features[p]
        rows.append(
            dict(
                ref=r,
                pred=p,
                iou=iou,
                voxels=int(pred[1][p]),
                reference_voxels=int(ref[1][r]),
                surface=len(inner),
                u=u,
            )
        )
    return rows


def fit(donors):
    groups = [rows for rows in donors.values() if rows]
    gram = 0.02 * np.eye(2)
    rhs = np.zeros(2)
    for rows in groups:
        x = np.array([[1.0, r["u"]] for r in rows])
        y = np.array([(r["reference_voxels"] - r["voxels"]) / r["surface"] for r in rows])
        w = 1.0 / (len(groups) * len(rows))
        gram += w * (x.T @ x)
        rhs += w * (x.T @ y)
    return np.linalg.solve(gram, rhs)


def correct(prob, features, coef):
    mask = prob >= 0.5
    changes = []
    for p, (inner, outer, u) in sorted(features.items()):
        delta = int(np.rint(len(inner) * np.clip(coef[0] + coef[1] * u, -0.25, 0.25)))
        cand = outer if delta > 0 else inner
        value = -prob.flat[cand] if delta > 0 else prob.flat[cand]
        take = cand[np.lexsort((cand, value))[: abs(delta)]]
        mask.flat[take] = delta > 0
        changes.append(
            dict(
                pred=p,
                surface=len(inner),
                outer=len(outer),
                u=u,
                requested=delta,
                applied=(1 if delta > 0 else -1) * len(take),
            )
        )
    return mask, changes


def measure(mask, truth, tx, i, folder, name):
    spacing = (SPEC[i][2],) * 3
    # Store the same hard mask at both slab sizes. Native calls classify <= threshold.
    table = folder / f"sample{i}-{name}-pores.csv"
    with TemporaryDirectory() as t:
        src = Path(t) / "mask.f32"
        (~mask).astype("<f4").tofile(src)
        stat = native(src, table, mask.shape, spacing, 0, 6, 1000, "none", 4)
        lp, rows = reference(mask, 6, spacing, table, stat)
        other = Path(t) / "second.csv"
        st2 = native(src, other, mask.shape, spacing, 0, 6, 37, "none", 64)
        if (stat["slab_depth"], st2["slab_depth"]) != (
            8,
            37,
        ) or table.read_bytes() != other.read_bytes():
            raise ValueError("Native slab mismatch")
    obj = objects(lp, truth, rows, folder / f"sample{i}-{name}-matches.csv")
    pred = label(mask)
    found = pairs(pred, tx, 64)
    if len(found) != obj["matches"]:
        raise ValueError("Independent match counts differ")
    m = score(mask, truth)
    eligible = np.flatnonzero(tx[2] & (tx[1] >= 64))
    ledger = {
        int(r): dict(
            reference_voxels=int(tx[1][r]),
            pred=found[r][0] if r in found else None,
            voxels=int(pred[1][found[r][0]]) if r in found else None,
            iou=found[r][1] if r in found else None,
        )
        for r in eligible
    }
    return dict(
        objects=obj,
        metrics=m,
        gates_on_research_slab=final_gates(m, obj),
        table_sha256=sha(table),
        mask_u8_sha256=digest(mask.astype("u1")),
        exact_components=len(rows),
        native=stat,
        slab_depths=[8, 37],
        references=ledger,
    )


def screen(base, arm, donors):
    b, a = [x["objects"] for x in (base, arm)]
    rr, ss = base["references"], arm["references"]
    if rr.keys() != ss.keys() or any(
        v["reference_voxels"] != ss[r]["reference_voxels"] for r, v in rr.items()
    ):
        raise ValueError("Different reference populations")
    ids = [
        k
        for k, v in rr.items()
        if 64 <= v["reference_voxels"] < 128
        and v["voxels"] is not None
        and ss[k]["voxels"] is not None
    ]
    tail = [
        float(
            np.percentile(
                [
                    100
                    * abs(d[k]["voxels"] - d[k]["reference_voxels"])
                    / d[k]["reference_voxels"]
                    for k in ids
                ],
                95,
            )
        )
        if ids
        else None
        for d in (rr, ss)
    ]
    missing = [x["eligible_reference"] - x["matches"] for x in (b, a)]
    unmatched = [x["eligible_prediction"] - x["matches"] for x in (b, a)]
    support = len(donors) == 2 and min(donors.values()) >= 5 and len(ids) >= 5
    allp = [x["p95_abs_volume_error_pct"] for x in (b, a)]
    checks = dict(
        support=support,
        shared_gain_ge3=bool(ids) and tail[0] - tail[1] >= 3,
        full_p95_not_worse=all(v is not None for v in allp) and allp[1] <= allp[0],
        missed_not_worse=missing[1] <= missing[0],
        unmatched_not_worse=unmatched[1] <= unmatched[0],
    )
    return dict(
        shared_ids=ids,
        shared_p95_pct=tail,
        full_p95_pct=allp,
        missed=missing,
        unmatched=unmatched,
        criteria=checks,
        decision="insufficient support"
        if not support
        else "pass"
        if all(checks.values())
        else "fail",
    )


def census(tx):
    _, sizes, w, _ = tx
    return dict(
        complete_32_63=int(sum(w & (sizes >= 32) & (sizes < 64))),
        complete_64_127=int(sum(w & (sizes >= 64) & (sizes < 128))),
        complete_ge128=int(sum(w & (sizes >= 128))),
        clipped_ge64=int(sum((~w[1:]) & (sizes[1:] >= 64))),
    )


def run(cache, baseline, out):
    manifest = json.loads((baseline / "baseline.json").read_text())
    if (
        set(manifest["specimens"]) != {str(i) for i in DEV}
        or manifest["z"] != list(Z)
        or manifest["dtype"] != "<f4"
        or manifest["shape_zyx"] != list(SHAPE)
    ):
        raise ValueError("Invalid baseline manifest")
    probs = {}
    for i in DEV:
        rec = manifest["specimens"][str(i)]
        src = baseline / f"sample{i}.f32"
        if sha(src) != rec["probability"] or src.stat().st_size != int(np.prod(SHAPE)) * 4:
            raise ValueError("Unsealed probability volume")
        pub = json.loads((ROOT / f"pores/evidence/candidate-e/sample{i}.json").read_text())
        if manifest["model"] != pub["pipeline_id"] or any(
            pub["source"][k]["sha256"] != rec[k] for k in ("raw", "segmented")
        ):
            raise ValueError("Different E/source identity")
        p = np.memmap(src, dtype="<f4", mode="r", shape=SHAPE)
        if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError("Invalid baseline range")
        probs[i] = p
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use an empty evidence directory")
    feats, library, truths = {}, {}, {}
    for i in DEV:
        p = probs[i]
        seg, _ = verified(cache, i, "segmented", False)
        truth = stack(seg, i, "segmented", *Z)
        pred = label(p >= 0.5)
        feats[i] = checked_features(p, pred)
        truths[i] = truth
        library[i] = donor_rows(pred, label(truth), feats[i])
    results = {}
    for i in DEV:
        donors = {k: v for k, v in library.items() if k != i}
        counts = {k: len(v) for k, v in donors.items()}
        coef = fit(donors)
        p, truth = probs[i], truths[i]
        base = p >= 0.5
        arm, changes = correct(p, feats[i], coef)
        tx = label(truth)
        b, a = [measure(mask, truth, tx, i, out, k) for mask, k in ((base, "base"), (arm, "arm"))]
        rec = dict(
            specimen=i,
            source_xyz=[360, 376, Z[0]],
            shape_zyx=list(SHAPE),
            sealed=manifest["specimens"][str(i)],
            reference_u8_sha256=digest(truth.astype("u1")),
            census=census(tx),
            cuda_faces=dict(
                ptx_sha256=sha(ROOT / "build/pores/transfer_faces.ptx"),
                inner=sum(len(t[0]) for t in feats[i].values()),
                outer=sum(len(t[1]) for t in feats[i].values()),
            ),
            donor_pairs_for_other_folds=library[i],
            trained_on=counts,
            coefficients=coef.tolist(),
            adjustments=changes,
            base=b,
            arm=a,
            screen=screen(b, a, counts),
        )
        (out / f"sample{i}.json").write_text(
            json.dumps(rec, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )
        np.savez_compressed(
            out / f"sample{i}-review.npz",
            probability=p,
            base=np.packbits(base),
            arm=np.packbits(arm),
            reference=np.packbits(truth),
            shape_zyx=SHAPE,
            origin_xyz=[360, 376, Z[0]],
        )
        print(i, rec["screen"], flush=True)
        results[i] = rec["screen"]["decision"]
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "baseline", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    run(a.cache, a.baseline, a.out)
