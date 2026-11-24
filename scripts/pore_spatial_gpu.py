"""Forward the small valid CNN using just the CUDA driver and an NVRTC-built kernel."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np


def forward(x, layers, ptx):
    x = np.ascontiguousarray(x, dtype="f4")
    nz, cin, h, w = x.shape
    if not layers or min(nz, cin, h, w) <= 0 or not np.isfinite(x).all():
        raise ValueError("Invalid CNN input")
    dll = ct.WinDLL("nvcuda")

    def call(name, *args):
        rc = getattr(dll, name)(*args)
        if rc:
            raise RuntimeError(f"{name}: CUDA error {rc}")

    with ExitStack() as scope:
        dev = ct.c_int()
        ctx, old, mod, fn = (ct.c_void_p() for _ in range(4))
        call("cuInit", 0)
        call("cuDeviceGet", ct.byref(dev), 0)
        call("cuDevicePrimaryCtxRetain", ct.byref(ctx), dev)
        scope.callback(call, "cuDevicePrimaryCtxRelease_v2", dev)
        call("cuCtxGetCurrent", ct.byref(old))
        call("cuCtxSetCurrent", ctx)
        scope.callback(call, "cuCtxSetCurrent", old)
        call("cuModuleLoadData", ct.byref(mod), ptx.read_bytes())
        scope.callback(call, "cuModuleUnload", mod)
        call("cuModuleGetFunction", ct.byref(fn), mod, b"spatial_conv")

        def allocate(n, arr=None):
            ptr = ct.c_uint64()
            call("cuMemAlloc_v2", ct.byref(ptr), ct.c_size_t(n))
            scope.callback(call, "cuMemFree_v2", ptr)
            if arr is not None:
                call("cuMemcpyHtoD_v2", ptr, ct.c_void_p(arr.ctypes.data), ct.c_size_t(n))
            return ptr

        src = allocate(x.nbytes, x)
        for ix, (weights, bias, d) in enumerate(layers):
            weights, bias = (np.ascontiguousarray(v, dtype="f4") for v in (weights, bias))
            co, ci, k, k2 = weights.shape
            if (
                k not in (1, 3)
                or k != k2
                or ci != cin
                or bias.shape != (co,)
                or d < 1
                or min(h, w) <= (k - 1) * d
            ):
                raise ValueError("Invalid layer shape")
            ow, oh = w - (k - 1) * d, h - (k - 1) * d
            ww = allocate(weights.nbytes, weights)
            bb = allocate(bias.nbytes, bias)
            out = allocate(nz * co * oh * ow * 4)
            args = [
                src,
                ww,
                bb,
                out,
                *[ct.c_int(v) for v in (w, h, ci, co, k, d, ix < len(layers) - 1)],
            ]
            argv = (ct.c_void_p * len(args))(*(ct.addressof(v) for v in args))
            call(
                "cuLaunchKernel",
                fn,
                (ow + 15) // 16,
                (oh + 7) // 8,
                nz * co,
                16,
                8,
                1,
                0,
                None,
                argv,
                None,
            )
            src, cin, h, w = out, co, oh, ow
        y = np.empty((nz, cin, h, w), dtype="f4")
        call("cuMemcpyDtoH_v2", ct.c_void_p(y.ctypes.data), src, ct.c_size_t(y.nbytes))
        return y
