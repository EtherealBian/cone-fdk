"""Reference-free reciprocal component transfer on two previously sealed masks."""

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pore_axial_raw import DEV, PTX, ROOT, SHAPE, Z, digest, sha, token, write
from pore_axial_score import read_mask
from pore_region import EV, read
from pore_region_gpu import launch
from pore_transfer import FACE, SPEC, label, native, reference
from scipy import ndimage as ndi

E = EV / "reciprocal-object"
PARENTS = (("local-profile", "profile"), ("census-train", "census"))
ARMS = ("forward", "reverse")


def identity():
    files = (
        "pores/RECIPROCAL-OBJECT-PILOT.md",
        "scripts/pore_reciprocal.py",
        "scripts/pore_reciprocal_score.py",
        "scripts/pore_region_gpu.py",
        "pores/region_medoid.cu",
        "scripts/pore_axial_score.py",
        "scripts/pore_transfer.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_section_score.py",
        "pores/validate.py",
        "pores/pore_check.py",
    )
    return dict(
        files={
            k: sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for k in files
        },
        parents={k: sha(EV / k / "predictions.json") for k, _ in PARENTS},
        compiled={k: sha(PTX / k) for k in ("region-medoid.ptx", "cone-pores.exe", "pores.ptx")},
    )


def parents(i):
    data = []
    src = []
    for k, a in PARENTS:
        d = read(EV / k / "predictions.json")
        assert d["shape_zyx"] == list(SHAPE) and d["origin_xyz"] == [360, 376, Z[0]]
        r = d["specimens"][str(i)]
        data.append(read_mask(EV / k, i, a, r["masks"][a]))
        src.append((r["raw"], r["segmented"]))
    assert src[0] == src[1]
    return data, *src[0]


def select(ms, spacing, out, prefix):
    reg = ms[0] | ms[1]
    tx = [label(m) for m in ms]
    labs = [t[0] for t in tx]
    u, _, _, boxes = label(reg)
    tables = []
    native_rows = []
    with TemporaryDirectory() as td:
        f = Path(td) / "input.f32"
        for name, m in zip(("profile", "census", "union"), (*ms, reg), strict=True):
            (~m).astype("<f4").tofile(f)
            table = out / f"{prefix}-{name}-native.csv"
            st = native(f, table, m.shape, spacing, 0, 6, 1000, "none", 4)
            _, rows = reference(m, 6, spacing, table, st)
            native_rows.append(rows)
            tables.append(sha(table))

    # Group native seeds by incidence in the independent flood below; no scored table is an input.
    def pos(r):
        return tuple(int(r["seed_" + k]) for k in "zyx")

    us = {int(u[pos(r)]): r for r in native_rows[2]}
    assert len(us) == len(boxes)
    w = np.zeros(len(boxes) + 1, "u1")
    outcpu = [m.copy() for m in ms]
    report = []
    for j, b in enumerate(boxes, 1):
        region = u[b] == j
        ls = [np.unique(field[b][region]) for field in labs]
        ls = [v[v != 0] for v in ls]
        cc = [[int(tx[a][1][q]) for q in qs] for a, qs in enumerate(ls)]
        ok = [len(v) == 1 for v in ls]
        h = int(np.sum(region & ms[0][b] & ms[1][b]))
        n = int(region.sum())
        reason = "replace"
        iou = None
        if not all(ok):
            reason = "component count"
        elif not all(bool(tx[a][2][v[0]]) for a, v in enumerate(ls)):
            reason = "clipped"
        elif min(cc[0][0], cc[1][0]) < 64:
            reason = "below 64"
        else:
            iou = Fraction(h, n)
            if iou < Fraction(1, 2):
                reason = "below IoU"
        w[j] = reason == "replace"
        # Second path: flood the binary union from its native seed (boxes only limit work),
        # then discover native component rows by their physical seed locations and verify sizes.
        ur = us[j]
        p = tuple(v - s.start for v, s in zip(pos(ur), b, strict=True))
        seed = np.zeros(region.shape, bool)
        seed[p] = True
        rr = ndi.binary_propagation(seed, structure=FACE, mask=reg[b])
        assert np.array_equal(rr, region) and int(rr.sum()) == int(ur["voxels"])
        rrsets = []
        for rows in native_rows[:2]:
            hits = []
            for row in rows:
                pt = pos(row)
                if (
                    all(s.start <= v < s.stop for v, s in zip(pt, b, strict=True))
                    and rr[tuple(v - s.start for v, s in zip(pt, b, strict=True))]
                ):
                    hits.append(row)
            rrsets.append(hits)
        assert all(
            sorted(map(lambda a: int(a["voxels"]), hits)) == sorted(vols)
            for hits, vols in zip(rrsets, cc, strict=True)
        )
        q = all(len(v) == 1 for v in rrsets) and all(
            int(v[0]["faces"]) == 0 and int(v[0]["voxels"]) >= 64 for v in rrsets
        )
        if q:
            a, bv = [int(v[0]["voxels"]) for v in rrsets]
            inter = a + bv - int(ur["voxels"])
            q = 2 * inter >= int(ur["voxels"])
            assert inter == h
        assert bool(q) == bool(w[j])
        changed = int(np.sum(region & (ms[0][b] ^ ms[1][b]))) if q else 0
        if q:
            for a in range(2):
                outcpu[a][b][rr] = ms[1 - a][b][rr]
        report.append(
            dict(
                region=j,
                union_seed=int(ur["id"]),
                area=n,
                component_ids=[[int(item["id"]) for item in hits] for hits in rrsets],
                sizes=cc,
                overlap=h,
                reason=reason,
                iou=str(iou) if iou is not None else None,
                changed_voxels=changed,
            )
        )
    codes = ms[0].astype("u1") + (ms[1].astype("u1") << 1)
    outs = []
    eligible = [int(np.sum(t[2] & (t[1] >= 64))) for t in tx]
    for a, ch in enumerate((w, 1 - w)):
        m = np.empty_like(reg)
        launch(b"copy_winner", (codes, u, ch), (m,), m.size)
        assert np.array_equal(m, outcpu[a])
        lt = label(m)
        assert int(np.sum(lt[2] & (lt[1] >= 64))) == eligible[a]
        outs.append(m)
    assert np.array_equal(outs[0] ^ outs[1], ms[0] ^ ms[1]) and np.array_equal(
        outs[0] & outs[1], ms[0] & ms[1]
    )
    return outs, dict(
        tables=tables,
        eligible=eligible,
        reasons=dict(Counter(v["reason"] for v in report)),
        regions=report,
    )


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    spec = {}
    ident = identity()
    for i in DEV:
        ms, raw, seg = parents(i)
        outs, g = select(ms, (SPEC[i][2],) * 3, out, f"sample{i}")
        ledger = out / f"sample{i}-choices.json"
        write(ledger, g)
        r = {}
        for k, m in zip(ARMS, outs, strict=True):
            f = out / f"sample{i}-{k}.npz"
            np.savez_compressed(f, mask=m)
            r[k] = dict(file=sha(f), mask=digest(m), cpu_gpu_differences=0)
        spec[str(i)] = dict(
            raw=raw, segmented=seg, parents=[digest(m) for m in ms], masks=r, ledger=sha(ledger)
        )
        print(i, g["reasons"], sum(v["changed_voxels"] for v in g["regions"]), flush=True)
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
    run(p.parse_args().out)
