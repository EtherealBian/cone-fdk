"""CPU file-to-file comparator for the standalone periodic-label study. Same dense ABI and fsync."""

import argparse
import ctypes
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from region_review import labels


def cpu(source, dest, h, w, k):
    if not (1 <= h <= 4096 and 3 <= w <= 4096 and 1 <= k <= 4096):
        raise ValueError("unsupported shape")
    if dest.exists():
        raise FileExistsError(dest)
    n = h * w
    ms = 0.0
    with source.open("rb") as fin:
        if os.fstat(fin.fileno()).st_size != k * n:
            raise ValueError("input size mismatch")
        with tempfile.NamedTemporaryFile(
            dir=dest.parent, prefix=dest.name + ".partial.", delete=False
        ) as fout:
            tmp = Path(fout.name)
        try:
            with tmp.open("wb") as fout:
                for _ in range(k):
                    a = np.frombuffer(fin.read(n), dtype="u1").reshape(h, w)
                    if a.max() > 2:
                        raise ValueError("invalid ternary byte")
                    t = time.perf_counter()
                    lab = labels(a == 1)
                    lab[a == 2] = -1
                    ms += (time.perf_counter() - t) * 1000
                    fout.write(lab.astype("<i4", copy=False).tobytes())
                fout.flush()
                os.fsync(fout.fileno())
            if os.name == "nt":
                move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
                move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
                move.restype = ctypes.c_int
                if not move(str(tmp), str(dest), 8):  # WRITE_THROUGH, no replacement
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                os.link(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
    return dict(cpu_label_ms=ms)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    for n in ("h", "w", "k"):
        p.add_argument(n, type=int)
    a = p.parse_args()
    print(json.dumps(cpu(**vars(a))))
