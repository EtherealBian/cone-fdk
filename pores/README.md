# Pore measurement lab — experimental, not a stable release

This branch adds native Windows x64 CUDA connectivity and component metrology for an **in-material rectangular ROI**. A separate research pipeline evaluates segmentation of NIST CoCr XCT TIFF stacks. They are already reconstructed scans, not ConeFDK projection inputs. The production tag remains the walnut reconstruction release until the [strict per-specimen gates](FINAL-GATE.md) pass.

[原始灰度剖面把七个共同低估孔都量进 ±15%，全体却未通过](RESULTS-FACE-PROFILE.md)

[锁定孔再量边界：没有丢匹配，但旧体积估计的理想范围仍不够](RESULTS-SHELL-VOLUME.md)

[三维灰度曲率：每件多找到一个孔，但同一批孔的体积误差都增加](RESULTS-VOLUME-CURVATURE.md)

[三个灰度能否区分孔和材料：每件至少判错多少体素，实际又漏了哪些孔？](RESULTS-BYTE-WINDOW.md)

[固定权重换到不重叠深度：三件体积误差 P95 均上升](RESULTS-CROSS-DEPTH.md)

[两套方法交换完整孔边界：正向不丢匹配，但只在 5 号通过改善筛选](RESULTS-RECIPROCAL-OBJECT.md)

[原图邻域梯度树：5 号找回 #62，3 号达到覆盖线，三件尚未同时达标](RESULTS-LOCAL-PROFILE.md)

[七组预测的区域选择：5 号实测 9/12，即使知道答案也最多 10/12](RESULTS-REGION-MEDOID.md) · [三个始终漏配的孔：位置偏差、粘连与小分量](RESULTS-PERSISTENT-MISS.md)

[12 项实验逐孔复核：5 号达到覆盖线时，已有预测的理想 P95 为 14.61%](RESULTS-OBJECT-CATALOG.md)

[八个方向：大多数误判并非朝向分歧](RESULTS-ORIENTATION.md) · [在相同边界上学习距离：优于只学符号，仍不及原基线](RESULTS-DISTANCE-TARGET.md)

[分组更新：能量确实降低，但比不改边界少匹配 15、11、3 个孔](RESULTS-MONOTONE.md)

[三维成像模型：三件都更能预测灰度，反推时却没有一个孔匹配](RESULTS-FORWARD.md)

[灰度积分筛选：挡住 20 个本来就小的孔，其中 14 个是弃判](RESULTS-MASS.md)

[小孔修复：找回 3 个，却新增 33 个未匹配预测](RESULTS-SMALL-SEED.md)

[五层三维解码：2 号同孔 P95 降至 10.96%，3、5 号变差](RESULTS-FIVE-PLANE.md) · [原始灰度：六个高对比度漏配孔的完整对照](RESULTS-LOCAL-CONTRAST.md)

[论文黑白插图与归档：20,502 个中心的独立重算](RESULTS-PRINTED-TARGET.md) · [双尺度 CUDA 对照：视野变大，匹配的孔没变](RESULTS-TWO-SCALE.md)

[局部连通监督：5 号九个孔量得更准，为何还不能晋级？](RESULTS-LOCAL-CONNECTIVITY.md) · [两步滤波的原图对照与三个试件的误差](RESULTS-FILTER-PROFILE.md)

[论文中间图定位：第 447 层的灰度实测与重放](RESULTS-FIGURE-GRAY.md) · [遍历校准体素：2 号同孔 P95 降至 14.89%，3、5 号未改善](RESULTS-CENSUS-TRAIN.md)

[相邻原始切片：找回一个被连到裁边的孔](RESULTS-AXIAL-RAW.md) · [窗口参数单位的配对检验](RESULTS-WINDOW-UNITS.md)

[灰度扰动上限：为什么 ±2 也找不回 3 号的三个孔](RESULTS-BYTE-RADIUS.md)

[截面面积监督：5 号同孔误差下降，却丢失一个 452 体素的对象](RESULTS-SECTION-AREA.md) · [历史 Fiji 阈值的逐字节对照](RESULTS-BERNSEN.md) · [原版 IPOL 去噪器](RESULTS-IPOL.md)

**[候选 H：共享权重、逐孔体积误差与三个漏配反例](RESULTS-H.md)** · [三件原始档重放与离线复核包](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-6)

**[候选 G：灰度校正和三种方法都判错的位置](RESULTS-G.md)** · [权重、原始切片与逐孔复核包](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-5)

**[候选 F 的逐孔对照与新切片的误拆反例](RESULTS-F.md)** · [三件离线复核包](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-4)

**[跨试件复核：三段新切片上的体积误差，以及两个参考孔为何被量成一个](RESULTS-TRANSFER.md)** · [下载三件离线复核包](https://github.com/EtherealBian/cone-fdk/releases/tag/boundary-transfer-pilot-1)

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

[Small-pore confidence-shell control](SHELL-PILOT.md): a separate cross-fitted experiment restricted to calibration data checks whether expanding or contracting only uncertain one-voxel boundaries improves small-pore measurement without adding misses or unmatched predictions. Its results do not change the E score.
