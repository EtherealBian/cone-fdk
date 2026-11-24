"""Compute disjoint inner/outer boundary ownership on the GPU (signed int32 labels)."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np


def owners(labels, eligible, ptx):
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    eligible = np.ascontiguousarray(eligible, dtype=np.uint8)
    if (
        labels.ndim != 3
        or labels.size == 0
        or labels.size > 2**31 - 1
        or labels.min() < 0
        or eligible.ndim != 1
        or len(eligible) <= labels.max()
        or eligible[0]
    ):
        raise ValueError("Invalid label volume/eligibility map")
    result = np.empty_like(labels)
    dll = ct.WinDLL("nvcuda")

    def call(name, *args):
        rc = getattr(dll, name)(*args)
        if rc:
            raise RuntimeError(f"{name}: CUDA error {rc}")

    with ExitStack() as scope:
        ctx, module, fn = (ct.c_void_p() for _ in range(3))
        dev = ct.c_int()
        call("cuInit", 0)
        call("cuDeviceGet", ct.byref(dev), 0)
        call("cuDevicePrimaryCtxRetain", ct.byref(ctx), dev)
        scope.callback(call, "cuDevicePrimaryCtxRelease_v2", dev)
        old = ct.c_void_p()
        call("cuCtxGetCurrent", ct.byref(old))
        call("cuCtxSetCurrent", ctx)
        scope.callback(call, "cuCtxSetCurrent", old)
        call("cuModuleLoadData", ct.byref(module), ptx.read_bytes())
        scope.callback(call, "cuModuleUnload", module)
        call("cuModuleGetFunction", ct.byref(fn), module, b"transfer_faces")
        ptrs = []
        for array in (labels, eligible, result):
            addr = ct.c_uint64()
            call("cuMemAlloc_v2", ct.byref(addr), ct.c_size_t(array.nbytes))
            scope.callback(call, "cuMemFree_v2", addr)
            if array is not result:
                call(
                    "cuMemcpyHtoD_v2",
                    addr,
                    ct.c_void_p(array.ctypes.data),
                    ct.c_size_t(array.nbytes),
                )
            ptrs.append(addr)
        args = ptrs + [ct.c_int(n) for n in reversed(labels.shape)]
        argv = (ct.c_void_p * len(args))(*(ct.addressof(a) for a in args))
        call(
            "cuLaunchKernel", fn, (labels.size + 255) // 256, 1, 1, 256, 1, 1, 0, None, argv, None
        )
        call(
            "cuMemcpyDtoH_v2",
            ct.c_void_p(result.ctypes.data),
            ptrs[-1],
            ct.c_size_t(result.nbytes),
        )
    return result
