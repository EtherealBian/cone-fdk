"""CUDA direct local gray residual; each launch visits one noninteracting residue class."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np
from pore_axial_raw import PTX
from pore_forward import kernel


def color_gpu(base, raw, q, phases):
    if (
        base.ndim != 3
        or base.dtype != bool
        or raw.dtype != np.uint16
        or base.shape != raw.shape
        or min(base.shape) < 9
        or base.size >= 2**31
        or not 0 <= phases[0] < phases[1] <= 131070
    ):
        raise ValueError("Invalid coordinate input")
    w, g = kernel(q)
    mask = np.ascontiguousarray(base).copy()
    gains = np.zeros(base.shape, "i8")
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
        call("cuModuleLoadData", ct.byref(mod), (PTX / "pore-monotone.ptx").read_bytes())
        clean.callback(call, "cuModuleUnload", mod)
        call("cuModuleGetFunction", ct.byref(fn), mod, b"coordinate")
        ptr = []
        for v in (base, mask, raw, w.astype("i4"), gains):
            a = np.ascontiguousarray(v)
            p = ct.c_uint64()
            n = ct.c_size_t(a.nbytes)
            call("cuMemAlloc_v2", ct.byref(p), n)
            clean.callback(call, "cuMemFree_v2", p)
            call("cuMemcpyHtoD_v2", p, ct.c_void_p(a.ctypes.data), n)
            ptr.append(p)
        args = (
            ptr
            + [ct.c_int(v) for v in (*reversed(base.shape), *phases)]
            + [ct.c_int64(int((g * g).sum())), ct.c_int(0)]
        )
        argv = (ct.c_void_p * len(args))(*(ct.addressof(v) for v in args))
        n = int(np.prod([(d + 4) // 5 for d in base.shape]))
        for color in range(125):
            args[-1].value = color
            call("cuLaunchKernel", fn, (n + 127) // 128, 1, 1, 128, 1, 1, 0, None, argv, None)
        for v, p in ((mask, ptr[1]), (gains, ptr[-1])):
            call("cuMemcpyDtoH_v2", ct.c_void_p(v.ctypes.data), p, ct.c_size_t(v.nbytes))
    return mask, gains
