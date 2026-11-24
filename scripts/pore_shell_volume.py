"""Detector-locked whole-object volume estimates; predictor accepts sealed masks, never labels."""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, PTX, ROOT, SHAPE, Z, digest, sha, token, write
from pore_axial_score import read_mask
from pore_region import EV, read
from pore_shell_gpu import fields
from pore_transfer import FACE, label
from scipy import ndimage as ndi

PANEL = (
    ("volume-curvature", "curvature"),
    ("volume-curvature", "profile"),
    ("census-train", "census"),
    ("byte-window", "source"),
    ("axial-raw", "axial"),
)
E = EV / "shell-volume"


def identity():
    files = (
        "pores/SHELL-VOLUME-PILOT.md",
        "pores/shell_volume.cu",
        "scripts/pore_shell_gpu.py",
        "scripts/pore_shell_volume.py",
        "scripts/pore_shell_volume_score.py",
        "scripts/pore_transfer.py",
        "scripts/pore_axial_score.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_object_catalog.py",
        "scripts/pore_section_score.py",
        "scripts/pore_compare_score_runs.py",
        "pores/validate.py",
        "pores/pore_check.py",
    )
    return dict(
        files={
            n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in files
        },
        panel=list(map(list, PANEL)),
        parents={f: sha(EV / f / "predictions.json") for f, _ in PANEL},
        compiled={n: sha(PTX / n) for n in ("shell-volume.ptx", "pores.ptx", "cone-pores.exe")},
    )


def parents(i):
    if i not in DEV:
        raise ValueError("Not a registered development specimen")
    ms, src = [], set()
    for f, k in PANEL:
        d = read(EV / f / "predictions.json")
        assert d["origin_xyz"] == [360, 376, Z[0]] and d["shape_zyx"] == list(SHAPE)
        r = d["specimens"][str(i)]
        src.add((r["raw"], r["segmented"]))
        ms.append(read_mask(EV / f, i, k, r["masks"][k]))
    assert len(src) == 1
    return ms, *src.pop()


def mutual(det, donor):
    a, sa, wa, _ = det
    b, sb, wb, _ = donor
    both = (wa & (sa >= 64))[a] & (wb & (sb >= 64))[b]
    key, n = np.unique(a[both].astype("i8") * len(sb) + b[both], return_counts=True)
    edges = []
    for h, v in zip(key, n, strict=True):
        j, k = divmod(int(h), len(sb))
        u = int(sa[j] + sb[k] - v)
        if 2 * int(v) >= u:
            edges.append((j, k, int(v), u))
    left, right = Counter(e[0] for e in edges), Counter(e[1] for e in edges)
    return {j: (k, v, u) for j, k, v, u in edges if left[j] == right[k] == 1}


def median(sizes):
    xs = sorted(sizes)
    n = len(xs)
    return (xs[n // 2] + xs[(n - 1) // 2] + 1) // 2


def project(owner, voters, target, face_ok):
    own = owner == 1
    codes = np.zeros(own.shape, "u1")
    support = codes.copy()
    for j, v in enumerate(voters):
        codes |= v.astype("u1") << j
        support += v
    deg = ndi.convolve(own.astype("u1"), FACE.astype("u1"), mode="constant", cval=0) - own
    foreign = ndi.binary_dilation(owner == 2, structure=FACE, iterations=2)
    pack = fields(owner, codes)
    assert np.array_equal(pack, support | (deg << 3) | (foreign.astype("u1") << 6))
    delta = int(target - own.sum())
    n = abs(delta)

    def candidates(s, d, b):
        if delta > 0:
            return ~own & (s > 0) & (d > 0) & ~b & face_ok
        return own & (s < len(voters)) & (d < 6)

    ix = np.flatnonzero(candidates(pack & 7, (pack >> 3) & 7, (pack & 64) != 0))
    cand = np.flatnonzero(candidates(support, deg, foreign))
    assert np.array_equal(ix, cand)
    if not n or len(ix) < n:
        return own, "equal" if not n else "insufficient shell", len(ix)
    sign = -1 if delta > 0 else 1
    keys = [sign * ((pack.ravel()[ix] >> t) & 7).astype("i2") for t in (0, 3)]
    order = ix[np.lexsort((ix, keys[1], keys[0]))][:n]
    # CPU cross-check: independent integer bins, then linear-index order inside ties.
    levels = [range(len(voters) + 1), range(7)]
    if delta > 0:
        levels = [reversed(levels[0]), tuple(reversed(levels[1]))]
    s, d = support.ravel()[cand], deg.ravel()[cand]
    other = np.concatenate([cand[(s == j) & (d == k)] for j in levels[0] for k in levels[1]])[:n]
    assert np.array_equal(order, other)
    new = own.copy()
    new.ravel()[order] = delta > 0
    if ndi.label(new, FACE)[1] != 1:
        return own, "disconnected", len(ix)
    assert new.sum() == target
    return new, "edited", len(ix)


def select(ms):
    if len(ms) != 5 or any(m.dtype != bool or m.ndim != 3 or m.shape != ms[0].shape for m in ms):
        raise ValueError("Expected five co-registered binary masks")
    det, *others = [label(m) for m in ms]
    a, sizes, whole, boxes = det
    out = ms[0].copy()
    ledger = {}
    checked = 0
    ed = [mutual(det, donor) for donor in others]
    expected = sizes.copy()
    for j in map(int, np.flatnonzero(whole & (sizes >= 64))):
        size = int(sizes[j])
        votes = []
        details = []
        vols = [size]
        box = tuple(
            slice(max(0, w.start - 3), min(n, w.stop + 3))
            for w, n in zip(boxes[j - 1], a.shape, strict=True)
        )
        loc = a[box]
        for k, (d, edge) in enumerate(zip(others, ed, strict=True)):
            if j not in edge:
                continue
            q, hit, union = edge[j]
            v = int(d[1][q])
            vols.append(v)
            vote = d[0][box] == q
            assert int((vote & (loc == j)).sum()) == hit and 2 * hit >= union == v + size - hit
            votes.append(vote)
            details.append(
                dict(arm=PANEL[k + 1][1], label=q, size=v, intersection=hit, union=union)
            )
        target = median(vols) if len(votes) >= 2 else size
        own = loc == j
        new = own
        reason = "few donors"
        nc = 0
        if len(votes) >= 2:
            face_ok = np.ones(loc.shape, bool)
            for ax, (w, n) in enumerate(zip(box, a.shape, strict=True)):
                plane = np.moveaxis(face_ok, ax, 0)
                if w.start == 0:
                    plane[0] = False
                if w.stop == n:
                    plane[-1] = False
            owner = own.astype("u1") + 2 * ((loc != 0) & ~own).astype("u1")
            new, reason, nc = project(owner, votes, target, face_ok)
            checked += owner.size
        coords = np.where(new != own)
        gi = np.ravel_multi_index(
            tuple(x + w.start for x, w in zip(coords, box, strict=True)), a.shape
        ).astype("<i8")
        assert np.array_equal(out[box][coords], own[coords])
        out[box][coords] = new[coords]
        expected[j] = int(new.sum())
        ledger[str(j)] = dict(
            detector=size,
            donors=details,
            target=target,
            delta=target - size,
            candidates=nc,
            reason=reason,
            actual=int(expected[j]),
            changed=len(gi),
            edit_indices_i64le=digest(gi),
        )
    # A bijection via shared original voxels, for ALL objects (even clipped/small), not just counts.
    nlab, ns, nw, _ = label(out)
    both = (a != 0) & out
    keys = np.unique(a[both].astype("i8") * len(ns) + nlab[both])
    src, dst = np.divmod(keys, len(ns))
    assert np.array_equal(src, np.arange(1, len(sizes))) and np.array_equal(
        np.sort(dst), np.arange(1, len(ns))
    )
    assert np.array_equal(ns[dst], expected[src]) and np.array_equal(nw[dst], whole[src])
    return out, dict(
        objects=ledger,
        all_components=len(keys),
        fields_checked=checked,
        reasons=dict(Counter(r["reason"] for r in ledger.values())),
    )


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    ident = identity()
    spec = {}
    for i in DEV:
        ms, raw, seg = parents(i)
        m, g = select(ms)
        f = out / f"sample{i}-shell.npz"
        np.savez_compressed(f, mask=m)
        ledger_file = out / f"sample{i}-choices.json"
        write(ledger_file, g)
        spec[str(i)] = dict(
            raw=raw,
            segmented=seg,
            parents=[digest(v) for v in ms],
            ledger=sha(ledger_file),
            masks=dict(shell=dict(file=sha(f), mask=digest(m))),
        )
        print(i, g["reasons"], g["fields_checked"], flush=True)
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            origin_xyz=[360, 376, Z[0]],
            shape_zyx=list(SHAPE),
            specimens=spec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args().out)
