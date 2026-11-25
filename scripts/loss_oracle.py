"""Independent CPU oracle: grow components from thresholded FP64 samples, not native labels."""

from collections import deque

import numpy as np


def cpu_labels(ref, part, threshold):
    h, w = ref.shape
    ok = (ref["status"] == 0) & (part["status"] == 0)
    mask = ok & (ref["thickness"] - part["thickness"] >= threshold)
    labels = np.where(ok, 0, -2 - 7 * ref["status"].astype(int) - part["status"].astype(int))
    for y, x in zip(*np.nonzero(mask), strict=True):
        if labels[y, x]:
            continue
        ident = int(y * w + x + 1)
        q = deque([(y, x)])
        labels[y, x] = ident
        while q:
            r, c = q.popleft()
            for rr, cc in ((r, (c - 1) % w), (r, (c + 1) % w), (r - 1, c), (r + 1, c)):
                if 0 <= rr < h and mask[rr, cc] and not labels[rr, cc]:
                    labels[rr, cc] = ident
                    q.append((rr, cc))
    return labels


def unpack(doc):
    target = np.zeros((doc["rows"], doc["columns"]), dtype=int)
    for row, x, length, ident in doc["runs"]:
        if not (
            0 <= row < target.shape[0]
            and 0 <= x
            and length > 0
            and x + length <= target.shape[1]
            and ident != 0
            and not target[row, x : x + length].any()
        ):
            raise ValueError("invalid or overlapping loss run")
        target[row, x : x + length] = ident
    return target
