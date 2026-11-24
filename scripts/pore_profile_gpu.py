"""Native numeric tree inference with bounds-checked exported model arrays."""

import ctypes as ct
from contextlib import ExitStack

import numpy as np
from pore_axial_raw import PTX


def pack(nodes, d):
    feat = []
    left = []
    right = []
    values = []
    roots = []
    start = 0
    for tree in nodes:
        leaf = tree["is_leaf"].astype(bool)
        internal = ~leaf
        index = np.flatnonzero(internal)
        assert len(tree) > 0 and not tree["is_categorical"].any()
        f = tree["feature_idx"][internal]
        assert np.all((f >= 0) & (f < d))
        for k in ("left", "right"):
            c = tree[k][internal]
            assert np.all((c > index) & (c < len(tree)))
        v = np.where(leaf, tree["value"], tree["num_threshold"])
        assert np.isfinite(v).all()
        roots.append(start)
        feat.extend(np.where(leaf, -1, tree["feature_idx"]))
        left.extend(np.where(leaf, 0, start + tree["left"]))
        right.extend(np.where(leaf, 0, start + tree["right"]))
        values.extend(v)
        start += len(tree)
    assert 0 < start < 2**31 and roots
    return tuple(
        np.asarray(v, dtype=dt)
        for v, dt in zip(
            (feat, left, right, values, roots), ("i4", "i4", "i4", "f8", "i4"), strict=True
        )
    )


def gpu_score(x, arrays, base):
    x = np.ascontiguousarray(x, dtype="f4")
    n, d = x.shape
    assert 0 < n < 2**31 and np.isfinite(x).all() and np.isfinite(base)
    out = np.empty(n, "f8")
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
        call("cuModuleLoadData", ct.byref(mod), (PTX / "local-profile.ptx").read_bytes())
        clean.callback(call, "cuModuleUnload", mod)
        call("cuModuleGetFunction", ct.byref(fn), mod, b"tree_score")
        ptr = []
        for a in (x, *arrays, out):
            p = ct.c_uint64()
            nb = ct.c_size_t(a.nbytes)
            call("cuMemAlloc_v2", ct.byref(p), nb)
            clean.callback(call, "cuMemFree_v2", p)
            if a is not out:
                call("cuMemcpyHtoD_v2", p, ct.c_void_p(a.ctypes.data), nb)
            ptr.append(p)
        args = [
            *ptr[:-1],
            ct.c_double(base),
            ptr[-1],
            ct.c_int(n),
            ct.c_int(d),
            ct.c_int(len(arrays[-1])),
        ]
        argv = (ct.c_void_p * len(args))(*(ct.addressof(v) for v in args))
        call("cuLaunchKernel", fn, (n + 255) // 256, 1, 1, 256, 1, 1, 0, None, argv, None)
        call("cuMemcpyDtoH_v2", ct.c_void_p(out.ctypes.data), ptr[-1], ct.c_size_t(out.nbytes))
    return out
