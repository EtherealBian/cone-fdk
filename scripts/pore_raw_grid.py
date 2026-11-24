"""Scanner-anchored raw box pyramid; independent CUDA, NumPy and per-site scalar paths."""

import ctypes as ct
from contextlib import ExitStack
from itertools import product
from math import floor

import numpy as np
from pore_axial_raw import PTX, SETTINGS

SCALES = {"fine": 1, "coarse": 4}
OFFSETS = tuple(product((-2, 0, 2), repeat=2))


def check(v, s):
    if v.dtype != np.uint16 or v.ndim != 3 or v.shape[0] != 3 or s not in (1, 4):
        raise ValueError("Expected three native uint16 planes; s=1 or 4")
    if any(k < 36 or k % 4 for k in v.shape[1:]):
        raise ValueError("Input must include real aligned halo")


def cpu_grid(v, s, i):
    check(v, s)
    a, b, c = v.astype("f8")
    f = np.stack([b, (a + c) / 2, abs(a - c)])
    h, w = b.shape
    cells = f.reshape(3, h // s, s, w // s, s).mean((2, 4))

    def coords(n):
        q = (np.arange(8, n - 8) + 0.5) / s - 0.5
        return q.astype(int), q % 1

    y, dy = coords(h)
    x, dx = coords(w)
    out = np.zeros((3, len(y), len(x)))
    for j, k in product(range(2), repeat=2):
        out += (
            cells[:, y[:, None] + j, x + k]
            * (dy if j else 1 - dy)[:, None]
            * (dx if k else 1 - dx)
        )
    gain, shift, _ = SETTINGS[i]
    out[:2] = ((out[:2] + shift) * gain - 128) / 48
    out[2] *= gain / 48
    return out.astype("f4")


def expand(grid, s):
    _, h, w = grid.shape
    return np.concatenate(
        [grid[:, 8 + j * s : h - 8 + j * s, 8 + k * s : w - 8 + k * s] for j, k in OFFSETS]
    )


def scalar(v, s, i, y, x):
    gain, shift, _ = SETTINGS[i]
    out = []
    for j, k in OFFSETS:
        qy = (16 + y + 0.5) / s - 0.5 + j
        qx = (16 + x + 0.5) / s - 0.5 + k
        by, bx = floor(qy), floor(qx)
        val = np.zeros(3)
        for yy, xx in product((by, by + 1), (bx, bx + 1)):
            weight = (1 - abs(yy - qy)) * (1 - abs(xx - qx))
            b = v[:, yy * s : (yy + 1) * s, xx * s : (xx + 1) * s].astype("i4")
            val += (
                weight
                * np.array([b[1].sum(), (b[0] + b[2]).sum() / 2, np.abs(b[0] - b[2]).sum()])
                / (s * s)
            )
        val[:2] = ((val[:2] + shift) * gain - 128) / 48
        val[2] *= gain / 48
        out.extend(val)
    return np.asarray(out)


class GridGPU:
    def __init__(self):
        self.scope = ExitStack()
        self.dll = ct.WinDLL("nvcuda")

    def call(self, name, *args):
        rc = getattr(self.dll, name)(*args)
        if rc:
            raise RuntimeError(f"{name}: CUDA error {rc}")

    def __enter__(self):
        d = ct.c_int()
        ctx, old, mod, self.fn = (ct.c_void_p() for _ in range(4))
        try:
            self.call("cuInit", 0)
            self.call("cuDeviceGet", ct.byref(d), 0)
            self.call("cuDevicePrimaryCtxRetain", ct.byref(ctx), d)
            self.scope.callback(self.call, "cuDevicePrimaryCtxRelease_v2", d)
            self.call("cuCtxGetCurrent", ct.byref(old))
            self.call("cuCtxSetCurrent", ctx)
            self.scope.callback(self.call, "cuCtxSetCurrent", old)
            self.call("cuModuleLoadData", ct.byref(mod), (PTX / "raw-grid.ptx").read_bytes())
            self.scope.callback(self.call, "cuModuleUnload", mod)
            self.call("cuModuleGetFunction", ct.byref(self.fn), mod, b"raw_grid")
            return self
        except BaseException:
            self.scope.close()
            raise

    def __exit__(self, *_):
        self.scope.close()

    def __call__(self, v, s, i):
        check(v, s)
        v = np.ascontiguousarray(v)
        _, h, w = v.shape
        out = np.empty((3, h - 16, w - 16), "f4")
        gain, shift, _ = SETTINGS[i]
        with ExitStack() as ex:
            p, q = ct.c_uint64(), ct.c_uint64()
            for ptr, n in [(p, v.nbytes), (q, out.nbytes)]:
                self.call("cuMemAlloc_v2", ct.byref(ptr), ct.c_size_t(n))
                ex.callback(self.call, "cuMemFree_v2", ptr)
            self.call("cuMemcpyHtoD_v2", p, ct.c_void_p(v.ctypes.data), ct.c_size_t(v.nbytes))
            args = [
                p,
                q,
                ct.c_int(w),
                ct.c_int(h),
                ct.c_int(s),
                ct.c_double(gain),
                ct.c_double(shift),
            ]
            argv = (ct.c_void_p * len(args))(*(ct.addressof(a) for a in args))
            self.call(
                "cuLaunchKernel",
                self.fn,
                (w - 16 + 15) // 16,
                (h - 16 + 7) // 8,
                1,
                16,
                8,
                1,
                0,
                None,
                argv,
                None,
            )
            self.call("cuMemcpyDtoH_v2", ct.c_void_p(out.ctypes.data), q, ct.c_size_t(out.nbytes))
        return out
