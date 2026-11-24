"""Exact integer curvature on raw native intensities, independent CPU and CUDA paths."""

import ctypes as ct
from contextlib import ExitStack
from itertools import product

import numpy as np
from pore_axial_raw import DEV, PTX, SETTINGS, Z

DIR = tuple(d for d in product((-1, 0, 1), repeat=3) if next((t for t in d if t), 0) > 0)


def check_raw(a):
    if a.ndim != 3 or a.dtype != np.uint16 or min(a.shape) < 11 or a.size * 15 >= 2**31:
        raise ValueError("Expected a bounded uint16 volume with five real halo samples")


def cpu(raw):
    check_raw(raw)
    a = raw.astype("i4")
    for axis in range(3):
        t = np.moveaxis(a, axis, 0)
        a = np.moveaxis(t[:-2] + t[1:-1] + t[2:], 0, axis)
    shape = tuple(n - 8 for n in a.shape)

    def offset(d):
        return a[tuple(slice(4 + v, 4 + v + n) for v, n in zip(d, shape, strict=True))]

    center = offset((0, 0, 0))
    out = []
    for r in (1, 2, 4):
        ds = [
            (offset(tuple(r * v for v in d)) + offset(tuple(-r * v for v in d)) - 2 * center)
            * (6 // sum(v * v for v in d))
            for d in DIR
        ]
        s = np.stack(ds, axis=-1)
        s.sort(axis=-1)
        out.append(s[..., ::3])
    return np.concatenate(out, axis=-1)


def scalar(raw, loc):
    def box(d, r):
        z, y, x = [5 + j + r * v for j, v in zip(loc, d, strict=True)]
        return int(raw[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2].sum(dtype="i8"))

    c = box((0, 0, 0), 0)
    out = []
    for r in (1, 2, 4):
        out.extend(
            sorted((box(d, r) + box(d, -r) - 2 * c) * (6 // sum(v * v for v in d)) for d in DIR)[
                ::3
            ]
        )
    return out


def gpu(raw):
    check_raw(raw)
    raw = np.ascontiguousarray(raw)
    z, y, x = raw.shape
    shape = tuple(n - 10 for n in raw.shape)
    out = np.empty((*shape, 15), "i4")
    dll = ct.WinDLL("nvcuda")

    def call(k, *args):
        rc = getattr(dll, k)(*args)
        if rc:
            raise RuntimeError(f"{k}: CUDA {rc}")

    with ExitStack() as clean:
        ctx, old, mod = (ct.c_void_p() for _ in range(3))
        dev = ct.c_int()
        call("cuInit", 0)
        call("cuDeviceGet", ct.byref(dev), 0)
        call("cuDevicePrimaryCtxRetain", ct.byref(ctx), dev)
        clean.callback(call, "cuDevicePrimaryCtxRelease_v2", dev)
        call("cuCtxGetCurrent", ct.byref(old))
        call("cuCtxSetCurrent", ctx)
        clean.callback(call, "cuCtxSetCurrent", old)
        call("cuModuleLoadData", ct.byref(mod), (PTX / "volume-curvature.ptx").read_bytes())
        clean.callback(call, "cuModuleUnload", mod)

        def alloc(n):
            p = ct.c_uint64()
            call("cuMemAlloc_v2", ct.byref(p), ct.c_size_t(n))
            clean.callback(call, "cuMemFree_v2", p)
            return p

        nb = (z - 2) * (y - 2) * (x - 2)
        p, q, o = alloc(raw.nbytes), alloc(4 * nb), alloc(out.nbytes)
        call("cuMemcpyHtoD_v2", p, ct.c_void_p(raw.ctypes.data), ct.c_size_t(raw.nbytes))
        for name, src, dest, num, delta in [
            (b"box27", p, q, nb, 0),
            (b"chords", q, o, out.size // 15, 2),
        ]:
            fn = ct.c_void_p()
            call("cuModuleGetFunction", ct.byref(fn), mod, name)
            args = [src, dest, *(ct.c_int(k - delta) for k in (z, y, x))]
            argv = (ct.c_void_p * 5)(*(ct.addressof(v) for v in args))
            call("cuLaunchKernel", fn, (num + 255) // 256, 1, 1, 256, 1, 1, 0, None, argv, None)
        call("cuMemcpyDtoH_v2", ct.c_void_p(out.ctypes.data), o, ct.c_size_t(out.nbytes))
    return out


def field(path, i, a, b):
    if i not in DEV or not any(m <= a < b <= n for m, n in ((160, 184), Z)):
        raise ValueError("Unregistered centers")
    from local_model import context_stack

    raw = context_stack(path, i, a - 4, b + 4, 5)
    assert raw.shape == (b - a + 10, 266, 266)
    x = gpu(raw)
    assert np.array_equal(x, cpu(raw))
    for loc in [(0, 0, 0), ((b - a) // 2, 128, 128), (b - a - 1, 255, 255)]:
        assert np.array_equal(x[loc], scalar(raw, loc))
    scale = np.repeat([SETTINGS[i][0] / (27 * 6 * r * r) for r in (1, 2, 4)], 5)
    return (x.astype("f8") * scale).astype("f4")
