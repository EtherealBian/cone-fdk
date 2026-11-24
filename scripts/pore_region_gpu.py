"""Small Windows CUDA Driver launch for the two registered region operators."""

import ctypes as ct
from contextlib import ExitStack

from pore_axial_raw import PTX


def launch(name, inputs, outputs, n):
    if name not in (b"votes", b"copy_winner") or not 0 < n < 2**31:
        raise ValueError("Invalid region launch")
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
        call("cuModuleLoadData", ct.byref(mod), (PTX / "region-medoid.ptx").read_bytes())
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
        args = [*ptr, ct.c_int(n)]
        argv = (ct.c_void_p * len(args))(*(ct.addressof(v) for v in args))
        call("cuLaunchKernel", fn, (n + 255) // 256, 1, 1, 256, 1, 1, 0, None, argv, None)
        for a, p in zip(outputs, ptr[len(inputs) :], strict=True):
            call("cuMemcpyDtoH_v2", ct.c_void_p(a.ctypes.data), p, ct.c_size_t(a.nbytes))
