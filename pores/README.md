# Pore measurement lab — experimental, not a stable release

This branch adds native Windows x64 CUDA connectivity and component metrology for an **in-material rectangular ROI**. A separate research pipeline evaluates segmentation of NIST CoCr XCT TIFF stacks. They are already reconstructed scans, not ConeFDK projection inputs. The production tag remains the walnut reconstruction release until the [strict per-specimen gates](FINAL-GATE.md) pass.

**[D 与 E：哪些小孔测得更准、哪些丢了？](RESULTS-E.md)** · **[E 离线复核包（原图与 C/D/E 同坐标）](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-3)**

**[C 与 D：逐孔配对、原始切片和全部失败项](RESULTS-D.md)** · **[D 离线复核包](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-2)** · **[基线和 A/B/C 的逐件结果](RESULTS.md)** · **[Offline numerical reviews and native mask preview](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-1)**

## Reproduce

Use Python 3.12 x64, the repository root `requirements-win.lock`, and an x64 MSVC Developer PowerShell or MinGW via `-Cxx`. From the repository root:

```powershell
.\pores\Build.ps1
.\.venv\Scripts\python.exe -m pytest -q tests pores/test_pores.py
# Protocol 1: verifies official SHA-256 digests before reading the scans.
.\.venv\Scripts\python.exe pores/validate.py --fetch
# Research candidates and full checks; these need the extra research dependencies.
.\.venv\Scripts\python.exe -m pip install -r pores/research-requirements.lock
.\.venv\Scripts\python.exe -m pytest -q tests pores
.\.venv\Scripts\python.exe pores/candidate.py --model C --fetch --specimens 2 3 5
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

Data: Felix H. Kim and coauthors, NIST, [10.18434/M32162](https://doi.org/10.18434/M32162), [original article](https://doi.org/10.1016/j.addma.2017.06.011). Data cropped, filtered, measured and annotated on 2026-11-23 and 2026-11-24. No NIST endorsement; see [notice](../NOTICE.md).

## Audit trail

[Baseline](evidence/baseline), [candidate A](evidence/candidate-a) and the separate [known-mask CPU/GPU comparison](evidence/mask-oracle) contain exact machine-readable reports and object tables. Sample 2 is a development scan; any samples inspected while changing a method are development scans for its successor. A successful same-mask check or high Dice on a development scan does **not** certify automatic segmentation. The binary validation target is the published digital reference, not independently measured physical dimensions.

For your own single-channel numbered TIFF slices, preserve source names, byte hashes, anisotropic spacing and ROI origin:

```powershell
.\.venv\Scripts\python.exe pores/import_stack.py C:\scans\part-A C:\scans\part-A-roi --xywh 360 376 256 256 --z 400 528 --spacing 2.45 2.45 2.45
```

This creates a new directory with `roi.f32` and `volume.json`. It does not assign a threshold or certify the input. Candidate B (`candidate.py --model B`) adds overlapping 5×5 patch estimates and uses guarded out-of-fold count and Dice calibration. Candidate C directly accumulates 5×5 and 7×7 patch distances in one native CUDA pass, adds local boundary curvature and calibrates two separately trained predictors with guard slices. D adds the six-output CUDA filter and adjacent-Z boundary evidence. E adds rank-adaptive patch estimates and balances both sides of complete small-pore boundaries during calibration. All five candidates are experimental. The classifier is research Python/scikit-learn; the standalone measurement executable is native C++/CUDA.

### Inspect mistakes rather than just the mean

After running a research candidate, execute `python pores/cases.py build/candidate-C 2 3 5` and `python pores/stratify.py build/candidate-C 2 3 5`. The case index covers **both** eligible component sets and jumps from mismatches to exact source coordinates. The size-band tables include absolute equivalent-diameter and centroid differences in µm; their small-sample percentiles are descriptive, not alternative release criteria. `python pores/audit.py build/candidate-C --development 2 3 5 --cache <archive-directory>` independently decodes the offline masks and checks all counts, pairings, object rows, digests, unrounded gates, source labels and every contrast-windowed raw review plane. It returns nonzero for a failed candidate.

The final numerical check requires the full independent pair, not a chosen subset: `python pores/audit.py <confirmation-output> --cache <archive-directory> --development-dir <dev-output>`. Passing this replay is necessary, not sufficient: the predictor must have been frozen before opening the two evaluation regions. A submitted development run cannot be converted to confirmation by changing a flag.

### Protect the confirmation specimens

The experimental snapshot predates the executable freeze lock and is development evidence only. The normal candidate CLI **refuses** `--specimens 4` or `6`. For a future candidate:

1. Re-run the frozen build on `--specimens 2 3 5`. `python pores/freeze.py <dev-output> --model C --cache <archive-directory> --out freeze.json` reloads the authenticated archives, checks the review planes and tables, and requires every gate on every development specimen. Failed runs cannot produce a manifest. The manifest binds the source, compiled measurement program, kernels, numerical software versions and exact development reports.
2. **Publish the manifest before accessing either confirmation ROI**; a local hash alone does not prove the order of events. Independently keep the timestamped commit. Then `python pores/candidate.py --model C --confirm freeze.json --development-dir <dev-output> --cache <archive-directory> --out <new-directory>` rechecks the evidence and runs 4 and 6 together. It will not overwrite previous confirmation output.
3. `python pores/audit.py <new-directory> --cache <archive-directory> --development-dir <dev-output>` replays both confirmation specimens and the original development evidence. Any failed specimen prevents acceptance. After inspection, the same holdouts cannot be described as unseen for a revised method.

### Reproduce the D diagnostics

After running `python pores/candidate.py --model D --specimens 2 3 5 --cache <archive-directory> --out build/candidate-D`, run `pores/cases.py`, `pores/stratify.py` and `scripts/pore_overlap.py` with arguments `build/candidate-D 2 3 5`. For the mechanism check, extract the three published C review ZIPs into `build/candidate-C` and run `python scripts/pore_cut.py build/candidate-C build/articulation-C --cache <archive-directory>`. `python scripts/pore_example.py build/candidate-C build/candidate-D sample3.png` renders the same four-panel raw-slice comparison. `python scripts/pore_report_d.py --check` checks the published comparison against the committed per-specimen records.

[校准片预处理对照](SUPPORT-PILOT.md)：用同一 CUDA 去噪算法检查 27、19 和 7 体素中值邻域。此对照不使用正式评估片段，不能代替逐孔验收。

### Reproduce the E diagnostics

Extract the previous C and D offline reviews into `build/candidate-C` and `build/candidate-D`. Use the same authenticated NIST archive directory for prediction and independent replay. Run the commands individually: the evaluator and auditor return nonzero for this published candidate because its development gates failed.

```powershell
python pores/candidate.py --model E --specimens 2 3 5 --cache <archive-directory> --out build/candidate-E
python pores/cases.py build/candidate-E 2 3 5
python pores/stratify.py build/candidate-E 2 3 5
python scripts/pore_overlap.py build/candidate-E 2 3 5
python scripts/pore_overlap.py build/candidate-D 2 3 5
python scripts/pore_pairs_de.py build/candidate-D build/candidate-E
python scripts/pore_review_queue.py build/candidate-C build/candidate-D build/candidate-E 2 3 5
python pores/audit.py build/candidate-E --development 2 3 5 --cache <archive-directory>
New-Item -ItemType Directory -Force build/rank-release | Out-Null
python scripts/pore_queue_bundle.py build/candidate-C build/candidate-D build/candidate-E build/rank-release 2 3 5
```

To check the committed tables without the scan archives, run `python scripts/pore_pairs_de.py pores/evidence/candidate-d pores/evidence/candidate-e --check` and `python scripts/pore_report_e.py --check`. The [sparse lattice](GRID-PILOT.md) and [orthogonal plane](PLANE-PILOT.md) calibration controls did not support changing the XY estimator. Confirmation specimens 4 and 6 remain reserved.
