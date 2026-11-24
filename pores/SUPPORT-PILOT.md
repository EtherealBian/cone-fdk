# First-stage median footprint: a calibration-only pilot

All A–D models start with a 27-voxel median. Can changing only this shared preprocessing assumption improve the simplest source-informed estimator? Compare the full cube (27) with face-and-edge (19) and face-only (7) 3-D medians. Keep the CUDA 5×5 non-local means estimator, Bernsen window, raw mapping, per-specimen contrast value and ROI fixed across the three arms. This does not claim to reproduce Avizo or choose between C and D.

Use development scans 2/3/5 only. Per arm/scan select a bandwidth (14/22/30/42 mapped units), contrast offset (−2/0/+2), threshold bias (−1/0/+1) by Dice on [160,168); tie-break in the order encoded in the script. Evaluate once on [176,184), leaving eight slices between blocks. Do not open the evaluation window or either reserved specimen. Promote a footprint as a standalone replacement for further investigation only if that same alternate increases check-block Dice by at least 0.005 over the cube on every development scan. This is a screening rule, not a change to any release gate. All arms and all checks must be published, irrespective of this decision.

## 结果：保留完整 27 体素邻域

独立校准片段上的 Dice（只用于筛选预处理，不是开发评估或最终验收）：

| 试件 | 完整 27 | 面与棱 19 | 仅面 7 |
|---|---:|---:|---:|
| 2 | 0.94458 | 0.94211 | 0.92588 |
| 3 | 0.95905 | 0.95432 | 0.94490 |
| 5 | 0.97852 | 0.97726 | 0.97271 |

两种替代方案在三个试件上的校准检查值都低于原方案，不进入整件验证。不能据此推断所有滤波器都一样，也不能把八张校准检查切片的 Dice 当成逐孔体积误差。见[所有参数、计数和结果](evidence/support-pilot.json)；复算：`python scripts/pore_support_pilot.py --cache <archive-directory> --out support-pilot.json`。
