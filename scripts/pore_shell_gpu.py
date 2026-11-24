"""Native object-local face/vote fields for detector-locked metrology."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np
from pore_axial_raw import PTX


def fields(owner, codes):
    if owner.shape != codes.shape or owner.ndim != 3 or not 0 < owner.size < 2**31:
        raise ValueError("Invalid shell volume")
    inputs = [np.ascontiguousarray(v, dtype="u1") for v in (owner, codes)]
    n = owner.size
    outputs = [np.zeros(owner.shape, "u1")]
    name = b"shell_fields"
    dll = ct.WinDLL("nvcuda")

    def call(k, *args):
        rc = getattr(dll, k)(*args)
        if rc:
            raise RuntimeError(f"{k}: CUDA {rc}")

    with ExitStack() as clean:
        ctx, old, mod, fn = (ct.c_void_p() for _ in range(4))
        dev = ct.c_int()
        call("cuInit", 0)
        call("cuDeviceGet", ct.byref(dev), 0)
        call("cuDevicePrimaryCtxRetain", ct.byref(ctx), dev)
        clean.callback(call, "cuDevicePrimaryCtxRelease_v2", dev)
        call("cuCtxGetCurrent", ct.byref(old))
        call("cuCtxSetCurrent", ctx)
        clean.callback(call, "cuCtxSetCurrent", old)
        call("cuModuleLoadData", ct.byref(mod), (PTX / "shell-volume.ptx").read_bytes())
        clean.callback(call, "cuModuleUnload", mod)
        call("cuModuleGetFunction", ct.byref(fn), mod, name)
        ptr = []
        for a in (*inputs, *outputs):
            assert a.flags.c_contiguous
            p = ct.c_uint64()
            nb = ct.c_size_t(a.nbytes)
            call("cuMemAlloc_v2", ct.byref(p), nb)
            clean.callback(call, "cuMemFree_v2", p)
            call("cuMemcpyHtoD_v2", p, ct.c_void_p(a.ctypes.data), nb)
            ptr.append(p)
        args = [*ptr, *(ct.c_int(j) for j in owner.shape)]
        argv = (ct.c_void_p * len(args))(*(ct.addressof(v) for v in args))
        call("cuLaunchKernel", fn, (n + 255) // 256, 1, 1, 256, 1, 1, 0, None, argv, None)
        for a, p in zip(outputs, ptr[len(inputs) :], strict=True):
            call("cuMemcpyDtoH_v2", ct.c_void_p(a.ctypes.data), p, ct.c_size_t(a.nbytes))

    return outputs[0]
