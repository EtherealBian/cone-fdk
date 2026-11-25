"""Borrow or copy one private V,A,U float buffer; never give ASTRA a mutable alias of the acquisition."""

from contextlib import contextmanager
import numpy as np


@contextmanager
def projection_lease(astra, pg, views, mode, on_registered=None):
    if (
        mode not in ("copy", "borrow")
        or not isinstance(views, np.ndarray)
        or views.ndim != 3
        or views.dtype != np.dtype("<f4")
        or min(views.shape) < 1
    ):
        raise ValueError("expected float32 A,V,U projection data and a known ownership mode")
    owner = np.array(views.transpose(1, 0, 2), dtype=np.float32, order="C", copy=True)
    pid = None
    try:
        pid = (astra.data3d.create if mode == "copy" else astra.data3d.link)("-sino", pg, owner)
        shared = astra.data3d.get_shared(pid)
        actual = shared.ctypes.data == owner.ctypes.data
        if (
            actual != (mode == "borrow")
            or shared.shape != owner.shape
            or shared.strides != owner.strides
            or shared.dtype != owner.dtype
        ):
            raise ValueError("unexpected ASTRA layout or ownership")
        if on_registered is not None:
            on_registered(actual, int(owner.nbytes))
        del shared
        if mode == "copy":
            del owner
        yield pid
    finally:
        if pid is not None:
            astra.data3d.delete(pid)
