# 把搜索窗口从 21×21 扩到 43×43，体素判定更准了吗？

**三件都没有。** 在已登记、与拟合不重叠的校准检查区，三件的 Dice、精确率与召回率均下降。没有拿表现最好的一件替换原流程，也没有进入正式逐孔评估。

|开发试件|对照 h → 加宽 h|Dice：对照 → 加宽|精确率变化|召回率变化|修复 / 新增假阳|修复 / 新增漏检|
|---|---:|---|---:|---:|---:|---:|
|2|30 → 20|.94405 → .92173|−.02314|−.02136|46 / 119|31 / 94|
|3|30 → 20|.95829 → .94188|−.01973|−.01324|261 / 861|516 / 922|
|5|35 → 20|.98131 → .97451|−.01160|−.00203|103 / 456|165 / 226|

四种候选搜索/补丁半径 × 六个带宽 × 三件，合计 72 组拟合记录。按精确分数选出的统一替代几何为 (21,2)，对照为 (10,2)，补丁均为 5×5；每种几何的带宽都只用四张拟合片选择。它检验的是**各自校准后**的几何，而不是固定 h 下的因果差值。5 号的总体孔隙率偏差反而从 −0.12207 降到绝对值仅 0.01502 个百分点；这个好看的总数掩盖了更多错判。

相同 h=30 的三张对照掩膜与历史公开记录逐位等价（核对摘要）。四种几何、三个含边缘的测试位置、各六个带宽，CUDA 输出与逐补丁 FP64 计算最大绝对差 0.001914 原生灰度单位以内。原手册的“former GPU mode”兼容性表实在 **PDF 第 7 页**，登记时误写为第 8 页；保留原文，不修改方法或结果。该表不能证明 NIST 当时使用的模式或版本。

[预登记及手册出处](WINDOW-UNITS-PILOT.md) · [评分前的 Git 提交](https://github.com/EtherealBian/cone-fdk/commit/b1780fa86c0416bf48aca68d9222645bfbd413bd) · [全部拟合、CPU/CUDA 数值和摘要](evidence/window-units/predictions.json) · [两臂掩膜及逐件转移计数](evidence/window-units)。每件检查 8×256×104 个体素。全八格转移表索引为 `4*参考 + 2*对照 + 加宽`（孔为 1）；并非三维对象计量。

```powershell
python scripts/ptx.py scripts/window_units.cu build/pores/window-units.ptx
python scripts/pore_window_units.py predict --cache $cache --out build/window-predict-new
python scripts/pore_window_units.py score --cache $cache --pred pores/evidence/window-units --out build/window-check-new.json
```
