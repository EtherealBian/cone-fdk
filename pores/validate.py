"""Reproduce the registered three-specimen NIST test. Python prepares/checks; native CUDA measures."""

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_check import ROOT, native, reference
from review import write_review
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
SPEC = {
    2: (980, 1010, 2.45),
    3: (988, 1013, 2.77),
    4: (984, 1010, 2.43),
    5: (984, 1010, 2.52),
    6: (984, 1013, 2.77),
}
ORIGIN = (360, 376, 400)
SOURCES = {
    s["filepath"]: s for s in json.loads((HERE / "sources.json").read_text(encoding="utf8"))
}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verified(cache, specimen, kind, fetch):
    stem = f"set1sample{specimen}{kind}"
    source = SOURCES[f"{kind}/{stem}.zip"]
    p = cache / (stem + ".zip")
    if specimen == 2 and not p.exists():
        alt = cache / ("raw.zip" if kind == "raw" else "seg.zip")
        if alt.is_file():
            p = alt
    if not p.exists() and fetch:
        tmp = p.with_suffix(".download")
        subprocess.run(
            ["curl.exe", "-fLsS", "--retry", "3", source["url"], "-o", str(tmp)], check=True
        )
        if sha(tmp) != source["sha256"]:
            raise ValueError(f"Invalid download: {tmp}")
        os.replace(tmp, p)
    if not p.is_file() or sha(p) != source["sha256"]:
        raise ValueError(f"Missing or corrupt archive: {p}; try --fetch")
    return p, source


def stack(path, specimen, kind, start, end):
    stem = f"set1sample{specimen}{kind}"
    slices = []
    with ZipFile(path) as archive:
        for z in range(start, end):
            with Image.open(io.BytesIO(archive.read(f"{stem}/{stem}_{z:04d}.tif"))) as im:
                if im.size != SPEC[specimen][:2]:
                    raise ValueError(f"Dimensions disagree: {im.size}")
                slices.append(np.array(im.crop((360, 376, 616, 632))))
    a = np.stack(slices)
    if kind == "segmented":
        if not np.isin(a, [0, 255]).all():
            raise ValueError("Expected binary NIST segmentation")
        return a == 0
    if a.dtype != np.uint16:
        raise ValueError("Expected uint16 intensity")
    return a


def calibration(filtered, truth):
    if not truth.any() or truth.all():
        raise ValueError("Calibration requires both phases")
    hits = np.bincount(filtered[truth], minlength=65536).cumsum()
    called = np.bincount(filtered.ravel(), minlength=65536).cumsum()
    return int(np.argmax(2 * hits / (int(truth.sum()) + called)))


def score(pred, ref):
    hit = int(np.count_nonzero(pred & ref))
    a = int(ref.sum())
    b = int(pred.sum())
    n = ref.size
    return dict(
        tp=hit,
        fp=b - hit,
        fn=a - hit,
        tn=n - a - b + hit,
        dice=2 * hit / (a + b) if a + b else 1.0,
        precision=hit / b if b else 0.0,
        recall=hit / a if a else 0.0,
        predicted_porosity_pct=100 * b / n,
        reference_porosity_pct=100 * a / n,
        porosity_error_pp=100 * (b - a) / n,
    )


def objects(lab, ref, rows, path):
    rlab, nr = ndi.label(ref)
    rc = np.bincount(rlab.ravel(), minlength=nr + 1)
    pc = np.bincount(lab.ravel(), minlength=len(rows) + 1)
    clipped = np.unique(
        np.concatenate([np.take(rlab, i, axis=a).ravel() for a in range(3) for i in (0, -1)])
    )
    rid = set(np.flatnonzero(rc >= 64)) - set(clipped) - {0}
    ids = {
        int(lab[int(r["seed_z"]), int(r["seed_y"]), int(r["seed_x"])]): int(r["id"]) for r in rows
    }
    pid = {
        int(lab[int(r["seed_z"]), int(r["seed_y"]), int(r["seed_x"])])
        for r in rows
        if int(r["faces"]) == 0 and int(r["voxels"]) >= 64
    }
    both = (lab > 0) & (rlab > 0)
    pairs, counts = np.unique(lab[both].astype("i8") * (nr + 1) + rlab[both], return_counts=True)
    candidates = sorted(
        [
            (float(c / (pc[p] + rc[r] - c)), int(p), int(r))
            for v, c in zip(pairs, counts, strict=True)
            for p, r in [divmod(int(v), nr + 1)]
            if p in pid and r in rid
        ],
        reverse=True,
    )

    def centers(labels, counts):
        coords = np.nonzero(labels)
        labels_fg = labels[coords]
        return np.stack(
            [
                np.bincount(labels_fg, weights=ax, minlength=len(counts)) / np.maximum(1, counts)
                for ax in coords
            ],
            axis=-1,
        )

    cp, cr = centers(lab, pc), centers(rlab, rc)
    refids = {}
    used = set()
    matches = []
    for iou, p, r in candidates:
        if iou < 0.5:
            break
        if p in used or r in refids:
            continue
        used.add(p)
        refids[r] = p
        matches.append(
            (
                ids[p],
                r,
                iou,
                int(pc[p]),
                int(rc[r]),
                100 * (float(pc[p]) / rc[r] - 1),
                100 * (np.cbrt(float(pc[p]) / rc[r]) - 1),
                float(np.linalg.norm(cp[p] - cr[r])),
            )
        )
    with path.open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pred_id",
                "ref_scipy_label",
                "iou",
                "pred_voxels",
                "ref_voxels",
                "signed_volume_error_pct",
                "signed_equiv_diameter_error_pct",
                "centroid_displacement_vox",
            ]
        )
        w.writerows(matches)
    return dict(
        criterion="both interior and >=64 voxels; descending IoU >=0.5, unique matches",
        eligible_prediction=len(pid),
        eligible_reference=len(rid),
        matches=len(matches),
        matched_fraction_prediction=len(matches) / len(pid) if pid else 0.0,
        matched_fraction_reference=len(matches) / len(rid) if rid else 0.0,
        **{
            k: float(np.percentile([abs(m[i]) for m in matches], q)) if matches else None
            for k, i, q in [
                ("median_abs_volume_error_pct", 5, 50),
                ("p95_abs_volume_error_pct", 5, 95),
                ("median_abs_diameter_error_pct", 6, 50),
                ("p95_centroid_displacement_vox", 7, 95),
            ]
        },
    )


def final_gates(m, o):
    return dict(
        dice=m["dice"] >= 0.95,
        precision=m["precision"] >= 0.93,
        recall=m["recall"] >= 0.93,
        porosity_absolute=abs(m["porosity_error_pp"]) <= 0.05,
        porosity_relative=m["reference_porosity_pct"] > 0
        and abs(m["porosity_error_pp"]) <= 0.05 * m["reference_porosity_pct"],
        object_prediction=o["matched_fraction_prediction"] >= 0.9,
        object_reference=o["matched_fraction_reference"] >= 0.9,
        **{
            k: o[k] is not None and o[k] <= limit
            for k, limit in [
                ("median_abs_volume_error_pct", 5),
                ("p95_abs_volume_error_pct", 15),
                ("median_abs_diameter_error_pct", 2),
                ("p95_centroid_displacement_vox", 1),
            ]
        },
    )


def run(specimen, cache, out, fetch):
    if not __debug__:
        raise RuntimeError("Do not use optimized Python; validation needs assertions")
    s = f"sample{specimen}"
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    sources = {k: verified(cache, specimen, k, fetch) for k in ("raw", "segmented")}

    def read(k, a, b):
        return stack(sources[k][0], specimen, k, a, b)

    def med(a):
        return ndi.median_filter(a, size=(1, 3, 3), mode="nearest")

    t = calibration(med(read("raw", 160, 184)), read("segmented", 160, 184))
    raw = read("raw", 400, 528)
    ref = read("segmented", 400, 528)
    filt = med(raw)
    pred = filt <= t
    spacing = (SPEC[specimen][2],) * 3
    table = out / (s + "-pores.csv")
    with TemporaryDirectory(dir=cache) as d:
        d = Path(d)
        src = d / "in.f32"
        raw.astype("<f4").tofile(src)
        a = native(src, table, raw.shape, spacing, t, 6, 1000, "median-xy", 4)
        lab, rows = reference(pred, 6, spacing, table, a)
        b = native(src, d / "b.csv", raw.shape, spacing, t, 6, 37, "median-xy", 64)
        if (a["slab_depth"], b["slab_depth"]) != (8, 37) or table.read_bytes() != (
            d / "b.csv"
        ).read_bytes():
            raise RuntimeError("Partition mismatch")
    metrics = score(pred, ref)
    m = metrics
    gates = dict(
        dice=m["dice"] >= 0.75,
        precision=m["precision"] >= 0.70,
        recall=m["recall"] >= 0.70,
        porosity=abs(m["porosity_error_pp"]) <= max(0.20, 0.15 * m["reference_porosity_pct"]),
    )
    agreement = objects(lab, ref, rows, out / (s + "-matches.csv"))
    viewer = out / (s + "-review.html")
    audit = write_review(viewer, specimen, raw, ref, lab, rows, ORIGIN, spacing[0])
    strict = final_gates(metrics, agreement)
    report = dict(
        protocol=1,
        final_gates=strict,
        final_passed=all(strict.values()),
        specimen=specimen,
        role="development" if specimen == 2 else "confirmation",
        source={k: v[1] for k, v in sources.items()},
        calibration_z=[160, 184],
        evaluation_z=[400, 528],
        origin_xyz=ORIGIN,
        threshold=t,
        metrics=metrics,
        baseline_gates=gates,
        baseline_passed=all(gates.values()),
        sensitivity={str(t + i): score(filt <= t + i, ref) for i in (-500, 500)},
        objects=agreement,
        table_sha256=sha(table),
        review=audit,
        review_sha256=sha(viewer),
        same_mask_scipy_rows=len(rows),
        second_slab_depth=b["slab_depth"],
        native=a,
    )
    (out / (s + ".json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(s, report["final_passed"], metrics, agreement, flush=True)
    return report["final_passed"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=ROOT / "data/pores")
    ap.add_argument("--out", type=Path, default=ROOT / "build/measurement")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--specimens", type=int, nargs="+", choices=[2, 3, 5], default=[2, 3, 5])
    args = ap.parse_args()
    results = [run(i, args.cache, args.out, args.fetch) for i in args.specimens]
    raise SystemExit(0 if all(results) else 1)
