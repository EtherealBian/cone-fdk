"""Exact integer partial-volume integrals, using native lattice distances on CUDA."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np
from pore_axial_raw import PTX


def terms(raw, obj, other):
    if (
        raw.dtype != np.uint16
        or raw.ndim != 3
        or obj.dtype != bool
        or other.dtype != bool
        or not raw.shape == obj.shape == other.shape
        or not 0 < obj.sum() <= 512
        or max(raw.shape) > 1024
    ):
        raise ValueError("Invalid native integration box")
    arrays = [
        np.ascontiguousarray(raw),
        np.ascontiguousarray(other, dtype="u1"),
        np.ascontiguousarray(np.argwhere(obj), dtype="i4"),
        np.zeros(5, "u8"),
    ]
    dll = ct.WinDLL("nvcuda")

    def call(name, *args):
        rc = getattr(dll, name)(*args)
        if rc:
            raise RuntimeError(f"{name}: CUDA {rc}")

    with ExitStack() as clean:
        dev = ct.c_int()
        ctx, old, mod, fn = (ct.c_void_p() for _ in range(4))
        call("cuInit", 0)
        call("cuDeviceGet", ct.byref(dev), 0)
        call("cuDevicePrimaryCtxRetain", ct.byref(ctx), dev)
        clean.callback(call, "cuDevicePrimaryCtxRelease_v2", dev)
        call("cuCtxGetCurrent", ct.byref(old))
        call("cuCtxSetCurrent", ctx)
        clean.callback(call, "cuCtxSetCurrent", old)
        call("cuModuleLoadData", ct.byref(mod), (PTX / "mass-terms.ptx").read_bytes())
        clean.callback(call, "cuModuleUnload", mod)
        call("cuModuleGetFunction", ct.byref(fn), mod, b"mass_terms")
        ptrs = []
        for arr in arrays:
            p = ct.c_uint64()
            n = ct.c_size_t(arr.nbytes)
            call("cuMemAlloc_v2", ct.byref(p), n)
            clean.callback(call, "cuMemFree_v2", p)
            call("cuMemcpyHtoD_v2", p, ct.c_void_p(arr.ctypes.data), n)
            ptrs.append(p)
        h, w = raw.shape[1:]
        a = ptrs + [ct.c_int(v) for v in (w, h, raw.size, len(arrays[2]))]
        argv = (ct.c_void_p * len(a))(*(ct.addressof(v) for v in a))
        call("cuLaunchKernel", fn, (raw.size + 127) // 128, 1, 1, 128, 1, 1, 0, None, argv, None)
        dest = arrays[-1]
        call("cuMemcpyDtoH_v2", ct.c_void_p(dest.ctypes.data), ptrs[-1], ct.c_size_t(dest.nbytes))
    return tuple(map(int, dest))
