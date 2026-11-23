# Pore measurement lab — experimental, not a stable release

This branch adds native Windows x64 CUDA connectivity and component metrology for an **in-material rectangular ROI**. A separate research pipeline evaluates segmentation of NIST CoCr XCT TIFF stacks. They are already reconstructed scans, not ConeFDK projection inputs. The production tag remains the walnut reconstruction release until the [strict per-specimen gates](FINAL-GATE.md) pass.

## Reproduce

Use Python 3.12 x64, the repository root `requirements-win.lock`, and an x64 MSVC Developer PowerShell or MinGW via `-Cxx`. From the repository root:

```powershell
.\pores\Build.ps1
.\.venv\Scripts\python.exe -m pytest -q tests pores
# Protocol 1: verifies official SHA-256 digests before reading the scans.
.\.venv\Scripts\python.exe pores/validate.py --fetch
# Experimental second algorithm; install only if needed.
.\.venv\Scripts\python.exe -m pip install -r pores/research-requirements.lock
.\.venv\Scripts\python.exe pores/candidate.py --fetch --specimens 2 3 5
```

Both evaluators intentionally exit **nonzero** if any selected specimen misses the final precision target. Subset runs cannot certify the full set. `--cache` selects the archive directory; `--out` selects the evidence directory. Every scan writes a CSV, one-to-one object matches, JSON and an offline HTML review. The review accepts every CSV ID and every Z index, shows the exact source TIFF name, and can display raw, native-verified component overlay, reference only or disagreements. All visual slices are contrast-windowed; the original intensities are not quantized for the baseline measurement. Reports include exact counts, not values rounded for display. For the experimental candidate, model inputs include 3D median + CUDA NLM features, per-scan supervised calibration slices and the published per-scan intensity mapping. **Not zero-label transfer.**

## Measure your own volume without Python

Pass little-endian finite FP32 voxels (x fastest). Use an internal crop or a validated mask; exterior dark air is indistinguishable from void. For a mask with 0=void and 1=material use `--threshold 0 --filter none`. Provide the three physical voxel spacings in micrometres. Example dimensions/spacing only:

```powershell
.\build\pores\cone-pores.exe --input roi.f32 --output objects.csv --nx 256 --ny 256 --nz 128 --sx 2.45 --sy 2.45 --sz 2.45 --threshold 0 --filter none --connectivity 6 --budget-mib 4
```

Uses `cone-pores.exe`, its sibling `pores.ptx` and the compatible NVIDIA driver (no cuFFT/Python needed at runtime). There are two 4-byte GPU buffers per current-slab voxel; the host component table can grow, and driver allocations are additional. 26-connectivity is also supported; do not mix it with six-connected benchmark results. `--filter median-xy` uses nearest XY boundaries. No general deconvolution, exterior exclusion or metrology certification is implied.

## Interpreting a row

`id` is one plus the first x-fastest linear voxel index in the ROI. `seed_*` is zero-based; add report `origin_xyz` to locate it in the original scan. Centroids use voxel centres in ROI-local µm. Equivalent diameter is the diameter of a sphere of the same volume (not a caliper diameter). `faces` bits 1/2/4/8/16/32 are x−/x+/y−/y+/z−/z+. Keep contacted components in total porosity, but do not report their volumes as complete pore volumes. Opposing-face contact is not a permeability measurement. Cross-segmentation object errors exclude clipped objects and objects under 64 voxels in **both** sets; missed and unmatched objects still count against match coverage.

Data: Felix H. Kim and coauthors, NIST, [10.18434/M32162](https://doi.org/10.18434/M32162), [original article](https://doi.org/10.1016/j.addma.2017.06.011). Data cropped, filtered, measured and annotated on 2026-11-23. No NIST endorsement; see [notice](../NOTICE.md).

## Audit trail

[Baseline](evidence/baseline), [candidate A](evidence/candidate-a) and the separate [known-mask CPU/GPU comparison](evidence/mask-oracle) contain exact machine-readable reports and object tables. Sample 2 is a development scan; any samples inspected while changing a method are development scans for its successor. A successful same-mask check or high Dice on a development scan does **not** certify automatic segmentation. The binary validation target is the published digital reference, not independently measured physical dimensions.

For your own single-channel numbered TIFF slices, preserve source names, byte hashes, anisotropic spacing and ROI origin:

```powershell
.\.venv\Scripts\python.exe pores/import_stack.py C:\scans\part-A C:\scans\part-A-roi --xywh 360 376 256 256 --z 400 528 --spacing 2.45 2.45 2.45
```

This creates a new directory with `roi.f32` and `volume.json`. It does not assign a threshold or certify the input. Candidate B (`candidate.py --model B`) adds overlapping 5×5 patch estimates and uses guarded out-of-fold count and Dice calibration. Both A and B are experimental.
