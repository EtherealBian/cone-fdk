# 一整片减薄在九组阈值下还连在一起吗？

带噪声的模拟内凹管给出了一个比“哪些点超限”更具体的答案：九组设置都只出现 **一片连通的减薄**。取 9.4 mm 为面积基准圆柱的半径，标称面积为 **8.630 mm²**；把一对阈值应用到整件后，实际达到的最小和最大面积分别为 **8.301 和 9.023 mm²**。这些是离散设置下的**圆柱投影足迹**，不是粗糙缺陷表面的真面积，也不是仪器误差带。

| 基准扫描边缘阈值 ↓ / 复检边缘阈值 →（/mm） | 0.018 | 0.020 | 0.022 |
|---|---:|---:|---:|
| **0.018** | 262 | 271 | 275 |
| **0.020** | 254 | **263** | 272 |
| **0.022** | 253 | 255 | 263 |

表内是超限单元数，减薄判据始终为 `基准壁厚 − 复检壁厚 >= 0.30 mm`。最小和最大各有一组明确的全局设置，而不是从不同设置拼出来的地图。每格投影面积为 `2π × 9.4 / 360 × 0.2 mm²`。[逐场景的峰值、坐标、面积与足迹标签哈希](region-noisy-change.json) · [与此前逐点阈值试验对照](EDGE-REVIEW.md)

## 如何确认是同一片？

只连接上下左右。方位列跨接 359° 和 0°，轴向首末行不接，两个对角也不接。每个区域以行优先的第一个超限单元索引加一编号；这是**本张地图内**的编号。跨设置比较请看 `nominal_overlap`，每条 `[标称区域 ID, 当前区域 ID, 共同单元数]` 表示两者在材料坐标上的实际交集。一对多是拆分，多对一是合并。编号恰好相同不能证明同一缺陷，未重叠也不证明缺陷消失。

局部不可测只影响用到该记录的阈值对。每组单独给出 `unknown_cells`；连通片邻接不可测区域和碰到轴向采样端面分别标注。不跨越盲区补洞。`observed_extrema` 包含所有场景**已观测到**的数量，`complete_grid_flagged_extrema` 只包含完整覆盖场景；没有这样的场景时为 `null`。原样复扫在九组阈值下均未报减薄。故意设错第二档时，四组中有三组覆盖不全，不能把观测为零解释成整件无缺陷。

## 不用 GPU 复算公开样例

在仓库根目录使用已装锁定依赖的 Python。用新目录；`$pin` 来自随仓库发布的独立验收记录。处理外来数据时应从可信渠道获取预期摘要。

```powershell
Expand-Archive reports/edge-review-example.zip -DestinationPath build/region-edge-example
$pin = (Get-Content reports/edge-qualification.json -Raw | ConvertFrom-Json).cases.'noisy-change'.review_manifest_sha256
python scripts/region_review.py build/region-edge-example build/region-result --area-radius-mm 9.4 --review-sha256 $pin
```

程序先检查摘要和记录，再重新枚举阈值对，逐格核对已有上下界和分类表。每组可查全图 int32 小端标签摘要（0 未超限、−1 不可测、正值为区域号）、区域峰值以及对应的原始材料行列。输出清单最后发布。退出 0 表示每组覆盖完整，2 表示结果有效但存在覆盖不全的设置，1 表示输入或计算未通过检查；都不是对工件质量的自动放行。算法有明确工作上限：最多 1,048,576 格、33,554,432 次配对单元、8,388,608 条输入壁记录；各组的区域和交集详单合计不超过 65,536 条。超限报错，不截断、不自动降采样。

## 验收证据

对原样复扫、带噪声内凹、无效边缘对照三组数据，独立运行原生 CUDA `loss`，并用另一套 CPU 洪泛算法从原始 `.wall` 记录重新连通。对 **22 张图、324,720 个配对单元**比对每一个标签，而不只比较区域总数；另核对峰值、最大值并列时的坐标、交集计数、面积和边界标记。小型反例覆盖接缝、禁止轴向回绕、禁止对角连接、阈值等号、桥接导致的拆分/合并、未知射线和输入篡改。[CUDA 对照与每张标签图的摘要](region-qualification.json)

```powershell
# 从独立模拟投影重新生成父作业和所有三层评定（见证、边缘、区域）。
python scripts/validate_witness.py --region-review --keep build/region-from-projections --reports build/region-from-projections/reports
# 或对已有三组边缘作业，指定另一套预期清单哈希，重跑原生/CPU 对照。
python scripts/validate_regions.py build/edge-final-replay build/region-replay --reports build/region-replay/reports --trusted reports/edge-qualification.json
```

完整重建回归的编译器、测试数量、512³ 核桃与 1296³ 压力体哈希在[隔离回归记录](region-regression.json)。
