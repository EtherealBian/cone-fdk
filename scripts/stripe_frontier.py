"""Axial-band 4-connected-component reducer. Only the previous frontier survives a band."""

import numpy as np
from scipy import ndimage


class Frontier:
    def __init__(self, width, anchors):
        if (
            type(width) is not int
            or width < 3
            or len(anchors) != 2
            or anchors[0] == anchors[1]
            or not all(
                len(p) == 2 and all(type(v) is int for v in p) and p[0] >= 0 and 0 <= p[1] < width
                for p in anchors
            )
        ):
            raise ValueError("invalid periodic width or anchors")
        self.w = width
        self.anchors = anchors
        self.row = 0
        self.last = np.zeros(width, dtype="i4")
        self.size = np.zeros(1, dtype="i8")
        self.flags = np.zeros(1, dtype="u1")
        self.finished = 0
        self.largest = 0
        self.linked = False
        self.hits = 0
        self.max_active = 0

    def feed(self, mask):
        if mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape[1] != self.w or not len(mask):
            raise ValueError("nonempty Boolean row band required")
        lab, n = ndimage.label(mask)
        k = len(self.size) - 1
        count = k + n + 1
        par = np.arange(count, dtype="i4")

        def root(i):
            while par[i] != i:
                par[i] = par[par[i]]
                i = par[i]
            return i

        def join(a, b):
            active = (a != 0) & (b != 0) & (a != b)
            for pair in np.unique(a[active].astype("i8") * count + b[active]):
                i, j = divmod(int(pair), count)
                i, j = root(i), root(j)
                if i != j:
                    par[max(i, j)] = min(i, j)

        counts = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        self.hits += int(counts.sum())
        marks = np.zeros(count, dtype="u1")
        marks[: k + 1] = self.flags
        # IDs 1..k belong to previously active components; local IDs start at k+1.
        lab = np.where(lab, lab + k, 0)
        for bit, (y, x) in enumerate(self.anchors):
            if self.row <= y < self.row + len(mask):
                local = lab[y - self.row, x]
                if local:
                    marks[local] |= 1 << bit
        join(lab[:, 0], lab[:, -1])
        join(self.last, lab[0])
        while True:
            shorter = par[par]
            if np.array_equal(shorter, par):
                break
            par = shorter
        masses = np.zeros(count, dtype="i8")
        np.add.at(masses, par[: k + 1], self.size)
        np.add.at(masses, par[k + 1 :], counts)
        flags = np.zeros(count, dtype="u1")
        np.bitwise_or.at(flags, par, marks)
        edge = par[lab[-1]]
        alive = np.unique(edge[edge != 0])
        retired = masses > 0
        retired[alive] = False
        self.finished += int(retired.sum())
        self.largest = max(self.largest, int(masses[retired].max(initial=0)))
        self.linked |= bool(np.any(flags == 3))
        remap = np.zeros(count, dtype="i4")
        remap[alive] = np.arange(1, len(alive) + 1, dtype="i4")
        self.last = remap[edge]
        self.size = np.r_[np.int64(0), masses[alive]]
        self.flags = np.r_[np.uint8(0), flags[alive]]
        self.row += len(mask)
        self.max_active = max(self.max_active, len(alive))

    def result(self):
        if self.row <= max(y for y, _ in self.anchors):
            raise ValueError("anchor outside processed axial extent")
        return dict(
            components=self.finished + len(self.size) - 1,
            largest_cells=max(self.largest, int(self.size.max(initial=0))),
            flagged_cells=self.hits,
            anchors_linked=self.linked,
        )
