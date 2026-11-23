# CoCr measurement acceptance — protocol 1 (2026-11-23)

Locked before evaluating new specimens. Scope: interior ROIs from the NIST Kim et al. CoCr LPBF XCT collection, DOI 10.18434/M32162. Sample 2 is development. Samples 3 and 5 are the two preselected confirmation specimens. Neither repeatability across scanners nor accuracy against physical ground truth is claimed.

For EACH specimen, use source XY [360,616) × [376,632); independent calibration slices [160,184) and evaluation slices [400,528), zero based TIFF indices. Per scan use the smallest uint16 threshold maximizing calibration Dice against the published segmentation, after a 3×3 XY nearest-edge median. Use <= and 6-connectivity. No threshold adjustment on evaluation data. Report perturbation ±500 intensity units and object agreement separately. No pooled-average pass.

1. Per specimen: evaluation Dice >=0.75, precision and recall each >=0.70, and absolute ROI porosity difference in percentage points <= max(0.20, 0.15 × reference porosity in percent). Use exact counts for gates.
2. Every native component ID, seed, count and face flags must match independent SciPy; all centroids, volumes and equivalent diameters must agree (rtol 1e-12). CSV bytes identical across 4 MiB / capped depth and 64 MiB / 37-plane runs. Native-to-CPU checks compare the same mask, not the NIST mask.
3. Every CSV row, including clipped and single-voxel objects, must resolve to its exact source TIFF, ROI coordinates and a non-empty selected-object overlay. Provide source digests, calibration/evaluation split, both masks, numerical CSV and offline review. Show object-level IoU and volume errors for context; no individual pore dimensional tolerance is claimed.
4. Report any failed specimen by name. Do not silently drop failures or change these gates. A changed method after inspection is a new protocol requiring fresh confirmation.
