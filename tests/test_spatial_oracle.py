"""Tiny independent float64 spatial-domain FDK oracle. No FFT, ASTRA or CUDA math.
Cropped detector and asymmetric signals exercise zero-extended interpolation at edges.
This is a correctness test only, not the production path or a timing baseline.
"""

import numpy as np
import pytest
from common import Geometry, reconstruct
from spatial_oracle import backproject_points


def oracle(projections, g, n, fov):
    z, y, x = (np.indices((n, n, n)) - (n - 1) / 2) * fov / n
    ref, border = backproject_points(projections, g, np.stack([x, y, z], -1))
    assert border, "test must exercise a fractional sample outside the detector"
    return ref


@pytest.mark.parametrize("signal,sign", [("edge-impulses", 1), ("random", -1)])
def test_direct_float64_oracle(tmp_path, signal, sign):
    scan = tmp_path / "oracle"
    g = Geometry(9, 7, 11, 0.67, 0.71, 92.0, 145.0, 0.213, sign)
    n = 13
    fov = 9.0
    g.save(scan.with_suffix(".geom"))
    p = np.random.default_rng(2311).normal(size=(g.a, g.v, g.u)).astype("<f4")
    if signal == "edge-impulses":
        p[:] = 0
        p[0, 0, 0] = 1.125
        p[4, -1, -1] = 0.75
        p[7, -1, 0] = -0.37
    p.tofile(scan.with_suffix(".f32"))
    out = tmp_path / "native.f32"
    reconstruct(scan, out, n, fov, 8, 4)
    ref = oracle(p.astype("f8"), g, n, fov)
    got = np.fromfile(out, "<f4").reshape(ref.shape)
    assert np.max(np.abs(ref)) > 0.001
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 5e-6
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-5

    # Same reference (including negative angles / clipped detector) but no full
    # volume in the native call. Asymmetric extents catch swapped axes.
    roi = tmp_path / "off-axis.f32"
    reconstruct(scan, roi, n, fov, 8, 4, roi="1,3,5,7,8,4")
    part = np.fromfile(roi, "<f4").reshape(4, 8, 7)
    expected = ref[5:9, 3:11, 1:8]
    assert np.linalg.norm(part - expected) / np.linalg.norm(expected) < 5e-6


@pytest.mark.parametrize("rows,cols", [(2, 2), (3, 8)])
def test_minimum_fft_workspace_and_detector_guards(tmp_path, rows, cols):
    # 2x2 exactly fills the reused FFT buffer. Batch=2 has a one-view tail;
    # the cropped FOV exercises both ends of each per-view table.
    g = Geometry(5, rows, cols, 1.1, 0.9, 50, 85, 0.32, 1)
    n = 19
    fov = 7.3
    p = np.random.default_rng(684).normal(size=(g.a, g.v, g.u)).astype("<f4")
    prefix = tmp_path / "tiny"
    g.save(prefix.with_suffix(".geom"))
    p.tofile(prefix.with_suffix(".f32"))
    out = tmp_path / "vol"
    m = reconstruct(prefix, out, n, fov, 4, 2)
    ref = oracle(p.astype("f8"), g, n, fov)
    got = np.fromfile(out, "<f4").reshape(ref.shape)
    assert m["redzones_verified"] and m["batch"] == 2
    assert np.linalg.norm(got - ref) / np.linalg.norm(ref) < 5e-6
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-5
