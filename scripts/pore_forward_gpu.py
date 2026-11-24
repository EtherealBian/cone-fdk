"""Native integer convolution of a binary mask and exact one-step decision."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np
from pore_axial_raw import PTX


def native(mask, weights, raw=None, phases=None):
    if mask.ndim != 3 or mask.dtype != bool or mask.size >= 2**31 or min(mask.shape) < 5:
        raise ValueError("Expected nonempty binary 3D volume")
    weights = np.asarray(weights)
    if (
        weights.shape != (5,)
        or weights.dtype.kind not in "iu"
        or np.any(weights < 0)
        or int(weights @ np.array([1, 6, 12, 8, 6])) != 98304
    ):
        raise ValueError("Wrong kernel normalization")
    if raw is not None and (
        raw.shape != mask.shape
        or raw.dtype != np.uint16
        or not 0 <= phases[0] < phases[1] <= 131070
    ):
        raise ValueError("Wrong native raw volume or phases")
    dll = ct.WinDLL("nvcuda")

    def call(name, *args):
        rc = getattr(dll, name)(*args)
        if rc:
            raise RuntimeError(f"{name}: {rc}")

    with ExitStack() as clean:
        dev = ct.c_int()
        ctx, old, mod = (ct.c_void_p() for _ in range(3))
        call("cuInit", 0)
        call("cuDeviceGet", ct.byref(dev), 0)
        call("cuDevicePrimaryCtxRetain", ct.byref(ctx), dev)
        clean.callback(call, "cuDevicePrimaryCtxRelease_v2", dev)
        call("cuCtxGetCurrent", ct.byref(old))
        call("cuCtxSetCurrent", ctx)
        clean.callback(call, "cuCtxSetCurrent", old)
        call("cuModuleLoadData", ct.byref(mod), (PTX / "pore-forward.ptx").read_bytes())
        clean.callback(call, "cuModuleUnload", mod)

        def alloc(arr, copy=True):
            a = np.ascontiguousarray(arr)
            n = ct.c_size_t(a.nbytes)
            p = ct.c_uint64()
            call("cuMemAlloc_v2", ct.byref(p), n)
            clean.callback(call, "cuMemFree_v2", p)
            if copy:
                call("cuMemcpyHtoD_v2", p, ct.c_void_p(a.ctypes.data), n)
            return p

        def launch(name, args):
            fn = ct.c_void_p()
            call("cuModuleGetFunction", ct.byref(fn), mod, name)
            argv = (ct.c_void_p * len(args))(*(ct.addressof(a) for a in args))
            call(
                "cuLaunchKernel",
                fn,
                (mask.size + 255) // 256,
                1,
                1,
                256,
                1,
                1,
                0,
                None,
                argv,
                None,
            )

        def download(a, p):
            call("cuMemcpyDtoH_v2", ct.c_void_p(a.ctypes.data), p, ct.c_size_t(a.nbytes))
            return a

        m = alloc(mask)
        w = alloc(weights.astype("i4"))
        field = np.empty(mask.shape, "i4")
        f = alloc(field, False)
        dims = [ct.c_int(v) for v in reversed(mask.shape)]
        launch(b"blur_occ", [m, w, f, *dims])
        if raw is None:
            return download(field, f)
        r = alloc(raw)
        out = np.empty_like(mask, order="C")
        q = alloc(out, False)
        p2, m2 = phases
        norm = int((weights.astype("i8") ** 2) @ np.array([1, 6, 12, 8, 6]))
        launch(
            b"inverse_step",
            [m, w, f, r, q, *dims, ct.c_int(m2), ct.c_int(m2 - p2), ct.c_int64(norm)],
        )
        return download(out, q)
