"""Compare the entire RTK-coordinate volume and measure the Release CLI on POPI."""

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import itk
from itk import RTK as rtk
import numpy as np

N = 256
PITCH = 1.25
LIMIT = 1e-3


def independent_fdk(folder):
    # Use the original geometry (not our parsed views, weights or matrices).
    geom = rtk.read_geometry(str(folder / "geometry.xml"))
    raw = np.memmap(folder / "views.f32", dtype="<f4", mode="c", shape=(644, 512, 512))
    proj = itk.image_view_from_array(raw)
    proj.SetSpacing((0.8, 0.8, 1.0))
    proj.SetOrigin((-204.4, -204.4, 0.0))
    seed = itk.image_from_array(np.zeros((N, N, N), dtype=np.float32))
    seed.SetOrigin((-(N - 1) * PITCH / 2,) * 3)
    seed.SetSpacing((PITCH,) * 3)
    image = type(seed)
    filt = rtk.FDKConeBeamReconstructionFilter[image].New()
    filt.SetInput(0, seed)
    filt.SetInput(1, proj)
    filt.SetGeometry(geom)
    filt.GetRampFilter().SetTruncationCorrection(0.0)
    filt.GetRampFilter().SetHannCutFrequency(0.0)
    filt.Update()
    return itk.array_from_image(filt.GetOutput())


def norm_diff(a, b):
    d = np.asarray(a, dtype=np.float64) - b
    return np.sqrt(np.sum(d * d, dtype=np.float64))


def verify(exe, folder):
    volume = folder / "volume.f32"
    cmd = [
        str(exe.resolve()),
        str(folder / "views.geom"),
        str(folder / "views.f32"),
        str(volume),
    ]
    times, kernels = [], []
    for rep in range(6):
        start = time.perf_counter_ns()
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = (time.perf_counter_ns() - start) * 1e-6
        key, val = p.stdout.strip().split("=")
        if key != "backproject_ms":
            raise ValueError("missing native CUDA event time")
        if rep:
            times.append(elapsed)
            kernels.append(float(val))
    ref = independent_fdk(folder)
    actual = np.fromfile(volume, "<f4").reshape(N, N, N)
    diff2 = ref2 = 0.0
    for a, b in zip(actual, ref, strict=True):
        diff2 += norm_diff(a, b) ** 2
        ref2 += np.sum(np.square(b, dtype=np.float64))
    rel = np.sqrt(diff2 / ref2)
    if not np.isfinite(rel) or rel > LIMIT:
        raise ValueError(f"full-volume relative L2 {rel:.8g} exceeds {LIMIT}")
    # Off-center slices in each of the published fixed axes must match in place,
    # not the opposite side, either in-plane mirror or a transposed slice.
    for axis in range(3):
        i = N // 3
        a, b = (np.take(v, i, axis=axis) for v in (actual, ref))
        alternatives = (np.take(ref, N - 1 - i, axis=axis), b[::-1], b[:, ::-1], b.T)
        if not all(norm_diff(a, b) < norm_diff(a, other) for other in alternatives):
            raise ValueError(f"slice orientation mismatch on storage axis {axis}")
    print("backproject_ms:", kernels, "median:", statistics.median(kernels))
    print(
        "cli_ms:",
        [round(t, 3) for t in times],
        "median:",
        round(statistics.median(times), 3),
    )
    print(f"full_volume_relative_l2={rel:.9g}; three slice orientations passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    verify(args.exe, args.folder)
