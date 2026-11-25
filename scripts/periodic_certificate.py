"""Verify dense component IDs against ternary evidence and an independent component count, in bands."""

import hashlib
import os
from pathlib import Path

import numpy as np
from anchor_audit import pin


def require(value, message):
    if not value:
        raise ValueError(message)


def same_neighbours(a, b):
    hits = (a > 0) & (b > 0)
    require(np.all(a[hits] == b[hits]), "adjacent measured cells have different IDs")


def certify(mask, labels, mask_sha256, shape, census, band_rows):
    """Census must come from an independent topology pass over the same evidence.

    Local adjacency implies constancy on each measured connected component.
    Matching the independent number of components makes the map onto IDs bijective.
    Self-witnessed IDs and ID <= row-major index+1 force the canonical minimum.
    None of these three conditions alone suffices. Never connect unknown cells.
    """
    from stripe_review import MAX_BAND_CELLS

    h, w = shape
    require(type(h) is int and type(w) is int and 1 <= h <= 4096 and 3 <= w <= 4096, "grid shape")
    require(
        type(band_rows) is int
        and 1 <= band_rows <= 4096
        and min(h, band_rows) * w <= MAX_BAND_CELLS,
        "band budget",
    )
    required = ("observed_components", "observed_flagged_cells", "unknown_cells")
    require(
        all(type(census[n]) is int and 0 <= census[n] <= h * w for n in required), "census range"
    )
    pin(mask_sha256)
    n = h * w
    seen = np.zeros(n + 1, "u1")  # per possible ID; no full-frame int32 labels held by verifier
    msha, lsha = hashlib.sha256(), hashlib.sha256()
    previous = None
    flagged = unknown = 0
    with Path(mask).open("rb") as mf, Path(labels).open("rb") as lf:
        require(
            os.fstat(mf.fileno()).st_size == n and os.fstat(lf.fileno()).st_size == 4 * n,
            "dense payload size",
        )
        for y in range(0, h, band_rows):
            count = min(band_rows, h - y) * w
            mb, lb = mf.read(count), lf.read(4 * count)
            msha.update(mb)
            lsha.update(lb)
            require(len(mb) == count and len(lb) == 4 * count, "short label evidence")
            m = np.frombuffer(mb, "u1").reshape(-1, w)
            v = np.frombuffer(lb, "<i4").reshape(-1, w)
            hit = m == 1
            blind = m == 2
            require(
                np.all(m <= 2)
                and np.all(v[m == 0] == 0)
                and np.all(v[blind] == -1)
                and np.all(v[hit] > 0),
                "ternary label support",
            )
            sites = np.flatnonzero(hit) + y * w + 1
            ids = v[hit]
            roots = ids == sites
            require(np.all(ids <= sites), "ID exceeds first possible cell")
            seen[ids] |= 1
            seen[ids[roots]] |= 2
            flagged += len(ids)
            unknown += int(blind.sum())
            same_neighbours(v[:, :-1], v[:, 1:])
            same_neighbours(v[:, -1], v[:, 0])
            same_neighbours(v[:-1], v[1:])
            if previous is not None:
                same_neighbours(previous, v[0])
            previous = v[-1].copy()
        require(not mf.read(1) and not lf.read(1), "trailing data")
    require(msha.hexdigest() == mask_sha256, "mask digest changed")
    require(not np.any(seen == 1), "ID has no measured root at its own cell")
    components = int(np.count_nonzero(seen))
    require(
        (components, flagged, unknown) == tuple(census[k] for k in required),
        "independent component or support census differs",
    )
    return dict(
        labels_i32le_sha256=lsha.hexdigest(),
        mask_u8_sha256=mask_sha256,
        observed_components=components,
        observed_flagged_cells=flagged,
        unknown_cells=unknown,
        verified_cells=n,
        verifier_id_bytes=seen.nbytes,
        peak_band_cells=min(h, band_rows) * w,
    )
