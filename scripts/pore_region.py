"""Label-free whole-region donor selection. No segmented source data or score files are read."""

import argparse
import hashlib
import io
import json
import zipfile
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pore_axial_raw import DEV, PTX, ROOT, SHAPE, Z, digest, sha, token, write
from pore_axial_score import read_mask
from pore_region_gpu import launch
from pore_transfer import SPEC, native, reference
from scipy import ndimage as ndi

PANEL = (
    ("section-area", "control"),
    ("axial-raw", "axial"),
    ("census-train", "census"),
    ("local-connectivity", "critical"),
    ("five-plane", "volume"),
    ("orientation", "mean8"),
    ("distance-target", "distance"),
)
EV = ROOT / "pores/evidence"
E = EV / "region-medoid"
ARMS = ("majority", "medoid")
BIT = np.array([[(c >> a) & 1 for a in range(7)] for c in range(128)], "i8")


def read(p):
    return json.loads(p.read_bytes())


def identity():
    files = (
        "pores/REGION-MEDOID-PILOT.md",
        "scripts/pore_region.py",
        "scripts/pore_region_gpu.py",
        "scripts/pore_region_score.py",
        "pores/region_medoid.cu",
        "scripts/pore_axial_score.py",
        "scripts/pore_section_score.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_transfer.py",
        "pores/validate.py",
        "pores/pore_check.py",
    )
    return dict(
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in files
        },
        parents={k + "/" + a: sha(EV / k / "predictions.json") for k, a in PANEL},
        compiled={k: sha(PTX / k) for k in ("region-medoid.ptx", "pores.ptx", "cone-pores.exe")},
    )


def scores(inter, sizes):
    scores = []
    for j in range(7):
        terms = []
        for k in range(7):
            u = int(sizes[j] + sizes[k] - inter[j, k])
            terms.append(Fraction(int(inter[j, k]), u) if u else Fraction(1))
        scores.append(sum(terms))
    return scores


def consensus(ms, lab):
    if (
        ms.dtype != bool
        or ms.ndim != 4
        or ms.shape[0] != 7
        or lab.shape != ms.shape[1:]
        or lab.dtype != np.int32
        or lab.min() < 0
        or not np.array_equal(lab != 0, ms.any(0))
    ):
        raise ValueError("Expected 7 bool masks and union labels")
    m = np.ascontiguousarray(ms)
    region = np.ascontiguousarray(lab)
    n = region.size
    count = int(region.max())
    h = np.zeros((count + 1, 128), "u4")
    codes = np.zeros(region.shape, "u1")
    major = np.zeros(region.shape, bool)
    med = np.zeros_like(major)
    launch(b"votes", (m, region), (codes, major, h), n)
    assert np.array_equal(major, np.count_nonzero(m, axis=0) >= 4) and np.array_equal(
        codes, np.packbits(m, axis=0, bitorder="little")[0]
    )
    winner = np.zeros(count + 1, "u1")
    other = np.zeros_like(med)
    ledger = []
    for j, box in enumerate(ndi.find_objects(region), 1):
        roi = region[box] == j
        v = m[(slice(None), *box)][:, roi]
        used = h[j].astype("i8")
        assert used.sum() == roi.sum() and not used[0]
        # Test every code bin independently, not just winner or the number of votes.
        truth_codes = np.sum(
            v.astype("u1") * (1 << np.arange(7, dtype="u1"))[:, None], axis=0, dtype="u1"
        )
        assert np.array_equal(np.bincount(truth_codes, minlength=128), used)
        sizes = used @ BIT
        inter = BIT.T @ (used[:, None] * BIT)
        s = scores(inter, sizes)
        independent = []
        for a in v:
            row = []
            for b in v:
                both = int(np.count_nonzero(a & b))
                union = int(np.count_nonzero(a | b))
                row.append(Fraction(both, union) if union else Fraction(1))
            independent.append(sum(row))
        assert s == independent
        tied = [k for k, a in enumerate(s) if a == max(s)]
        w = tied[0]
        winner[j] = w
        other[box][roi] = v[independent.index(max(independent))]
        ledger.append(
            dict(
                region=j,
                seed_zyx=[
                    int(c + sl.start)
                    for c, sl in zip(
                        np.unravel_index(np.argmax(roi), roi.shape), box, strict=True
                    )
                ],
                area=int(roi.sum()),
                donor_sizes=sizes.tolist(),
                nonempty_code_counts={int(c): int(used[c]) for c in np.flatnonzero(used)},
                jaccard_sums=[str(a) for a in s],
                tied=tied,
                chosen=w,
            )
        )
    launch(b"copy_winner", (codes, region, winner), (med,), n)
    assert np.array_equal(med, other)
    return major, med, ledger


def donor(i, k, a):
    folder = EV / k
    doc = read(folder / "predictions.json")
    r = doc["specimens"][str(i)]
    rec = r["masks"][a]
    assert doc["shape_zyx"] == list(SHAPE) and doc["origin_xyz"] == [360, 376, Z[0]]
    if k == "section-area":
        with zipfile.ZipFile(folder / "weights-masks.zip") as zipf:
            buf = zipf.read(f"sample{i}-{a}.npy")
        assert hashlib.sha256(buf).hexdigest() == rec["file"]
        m = np.load(io.BytesIO(buf), allow_pickle=False)
        assert m.dtype == bool and m.shape == SHAPE and digest(m) == rec["mask"]
    else:
        m = read_mask(folder, i, a, rec)
    return m, r["raw"], r["segmented"]


def run(out):
    ident = identity()
    spec = {}
    out.mkdir(parents=True, exist_ok=False)
    for i in DEV:
        data = [donor(i, k, a) for k, a in PANEL]
        assert len({(raw, seg) for _, raw, seg in data}) == 1
        ms = np.stack([x[0] for x in data])
        u = ms.any(axis=0)
        tab = out / f"sample{i}-union-pores.csv"
        spacing = (SPEC[i][2],) * 3
        with TemporaryDirectory() as tmp:
            f = Path(tmp) / "union.f32"
            (~u).astype("<f4").tofile(f)
            st = native(f, tab, u.shape, spacing, 0, 6, 1000, "none", 4)
            lab, rows = reference(u, 6, spacing, tab, st)
        major, med, ledger = consensus(ms, lab)
        lr = out / f"sample{i}-regions.json"
        write(lr, dict(panel=PANEL, regions=ledger))
        masks = {}
        for k, m in zip(ARMS, (major, med), strict=True):
            p = out / f"sample{i}-{k}.npz"
            with p.open("xb") as fp:
                np.savez_compressed(fp, mask=m)
            masks[k] = dict(mask=digest(m), file=sha(p), cpu_gpu_differences=0)
        spec[str(i)] = dict(
            raw=data[0][1],
            segmented=data[0][2],
            donor_masks=[digest(x) for x in ms],
            masks=masks,
            regions=len(rows),
            ledger_sha256=sha(lr),
            union_table=sha(tab),
            union_mask=digest(u),
            majority_medoid_differences=int(np.sum(major != med)),
        )
        print(i, len(rows), spec[str(i)]["majority_medoid_differences"], flush=True)
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            shape_zyx=list(SHAPE),
            origin_xyz=[360, 376, Z[0]],
            specimens=spec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.out)
