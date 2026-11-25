"""Integer cell-face ledger for a certified defect map. No interpolation or physical perimeter."""

import hashlib
from itertools import zip_longest
from pathlib import Path

import numpy as np
from periodic_certificate import require

MAX_COMPONENTS = 32768


def capacity(census):
    n = census["observed_components"]
    require(
        type(n) is int and 0 <= n <= MAX_COMPONENTS, "component ledger exceeds supported count"
    )
    return n


DIRECTIONS = ("north", "east", "south", "west")
KINDS = ("measured_below", "unknown", "axial_cut")


def windows(f, h, w, rows, dtype, outside, audit):
    """Read each source row exactly once while exposing an axial halo and bounded core."""
    dt = np.dtype(dtype)
    before = np.full((1, w), outside, dt)
    carry = None

    def take(n):
        b = f.read(n * w * dt.itemsize)
        require(len(b) == n * w * dt.itemsize, "truncated interface source")
        audit.update(b)
        return np.frombuffer(b, dt).reshape(n, w)

    for y in range(0, h, rows):
        nr = min(rows, h - y)
        core = take(nr) if carry is None else np.concatenate((carry, take(nr - 1)))
        after = take(1) if y + nr < h else np.full((1, w), outside, dt)
        yield np.concatenate((before, core, after))
        before = core[-1:].copy()
        carry = after
    require(not f.read(1), "trailing interface source")


def neighbours(padded):
    m = padded[1:-1]
    return padded[:-2], np.roll(m, -1, axis=1), padded[2:], np.roll(m, 1, axis=1)


def codes(halo):
    """A separate NumPy encoder; callers may pass 3 only for top/bottom halo rows."""
    core = halo[1:-1]
    require(core.size and np.all(core <= 2) and np.all(halo <= 3), "invalid ternary plane")
    out = np.zeros(core.shape, np.uint8)
    lut = np.array([1, 0, 2, 3], np.uint8)
    for d, v in enumerate(neighbours(halo)):
        out |= lut[v] << (2 * d)
    out[core != 1] = 0
    return out


def ledger(mask, labels, face, shape, expected, rows):
    """Independently compare every face byte, then reduce under certified global component IDs."""
    h, w = shape
    require(
        type(h) is int
        and type(w) is int
        and 1 <= h <= 4096
        and 3 <= w <= 4096
        and type(rows) is int
        and 1 <= rows <= 4096
        and min(rows, h) * w <= 1 << 20,
        "interface band or shape",
    )
    a, b, c = (hashlib.sha256() for _ in range(3))
    limit = capacity(expected)
    tables = {}
    observed = 0
    with Path(mask).open("rb") as mf, Path(labels).open("rb") as lf, Path(face).open("rb") as ff:
        for m, lab in zip_longest(
            windows(mf, h, w, rows, "u1", 3, a), windows(lf, h, w, rows, "<i4", -2, b)
        ):
            require(m is not None and lab is not None, "misaligned sources")
            known = codes(m)
            raw = ff.read(known.size)
            require(raw == known.tobytes(), "interface byte does not match source neighbors")
            c.update(raw)
            source = m[1:-1]
            tag = lab[1:-1]
            hit = source == 1
            require(
                np.all(tag[source == 0] == 0)
                and np.all(tag[source == 2] == -1)
                and np.all(tag[hit] > 0),
                "label support",
            )
            observed += int(hit.sum())
            ids, inv = np.unique(tag[hit], return_inverse=True)
            if not ids.size:
                continue

            def count(select, inverse=inv, n=len(ids)):
                return np.bincount(inverse[select], minlength=n)

            chunk = np.zeros((len(ids), 14), dtype=np.int64)
            chunk[:, 0] = np.bincount(inv, minlength=len(ids))
            measured_labels = tag[hit]
            packed = known[hit]
            for d, adj in enumerate(neighbours(lab)):
                nm = adj[hit]
                positive = nm > 0
                require(
                    np.all(nm[positive] == measured_labels[positive]),
                    "adjacent positive IDs disagree",
                )
                if d in (1, 2):
                    chunk[:, 1] += count(
                        positive
                    )  # count each shared E/S side once, seam included
                for k in (1, 2, 3):
                    chunk[:, 2 + 3 * d + k - 1] = count(((packed >> (2 * d)) & 3) == k)
            for ident, row in zip(ids, chunk, strict=True):
                key = int(ident)
                if key not in tables:
                    require(len(tables) < limit, "component table exceeds certified census")
                    tables[key] = row.copy()
                else:
                    tables[key] += row
        require(not ff.read(1), "trailing interface bytes")
    require(
        a.hexdigest() == expected["mask_u8_sha256"]
        and b.hexdigest() == expected["labels_i32le_sha256"],
        "different mask or canonical labels",
    )
    require(
        len(tables) == expected["observed_components"]
        and observed == expected["observed_flagged_cells"],
        "independent census mismatch",
    )
    components = []
    grand = np.zeros((4, 3), dtype=np.int64)
    for key, t in sorted(tables.items()):
        area, inside = map(int, t[:2])
        faces = t[2:].reshape(4, 3)
        require(
            4 * area == 2 * inside + int(faces.sum()) and faces[1, 2] == faces[3, 2] == 0,
            "cell-face conservation or axial clipping mismatch",
        )
        grand += faces
        components.append(
            dict(id=key, area_cells=area, internal_undirected_edges=inside, faces=faces.tolist())
        )
    return dict(
        face_u8_sha256=c.hexdigest(),
        directions=list(DIRECTIONS),
        face_kinds=list(KINDS),
        components=components,
        faces_by_direction_and_kind=grand.tolist(),
        observed_cells=observed,
        source_cells=h * w,
    )
