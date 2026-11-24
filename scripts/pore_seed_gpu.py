"""Native six-neighbor median on real raw halo; no normalized floats or ties to resolve."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np
from pore_axial_raw import PTX


def median7(raw):
    if raw.ndim != 3 or raw.dtype != np.uint16 or min(raw.shape) <= 8 or raw.size >= 2**31:
        raise ValueError("Expected uint16 (z,y,x) with a four-voxel margin")
    raw = np.ascontiguousarray(raw)
    out = np.empty(tuple(n - 8 for n in raw.shape), "u2")
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
        call("cuModuleLoadData", ct.byref(mod), (PTX / "seed-median.ptx").read_bytes())
        clean.callback(call, "cuModuleUnload", mod)
        call("cuModuleGetFunction", ct.byref(fn), mod, b"seed_median7")
        p, q = ct.c_uint64(), ct.c_uint64()
        for ptr, n in [(p, raw.nbytes), (q, out.nbytes)]:
            call("cuMemAlloc_v2", ct.byref(ptr), ct.c_size_t(n))
            clean.callback(call, "cuMemFree_v2", ptr)
        call("cuMemcpyHtoD_v2", p, ct.c_void_p(raw.ctypes.data), ct.c_size_t(raw.nbytes))
        args = [p, q, *(ct.c_int(n) for n in reversed(out.shape))]
        argv = (ct.c_void_p * len(args))(*(ct.addressof(a) for a in args))
        call("cuLaunchKernel", fn, (out.size + 255) // 256, 1, 1, 256, 1, 1, 0, None, argv, None)
        call("cuMemcpyDtoH_v2", ct.c_void_p(out.ctypes.data), q, ct.c_size_t(out.nbytes))
    return out
