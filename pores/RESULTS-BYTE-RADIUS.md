# 只改动一两个灰度级，漏掉的小孔能找回来吗？

对固定 h=30 的 CUDA 去噪与严格 Bernsen 基线，**3 号开发试件不能**。即使提前知道 NIST 数字参考、允许每个位置都选最有利的局部改动，±2 个 8 位灰度级也最多能匹配 24/27 个合格参考孔（88.89%，要求至少 25 个）。这排除的是该基线的有界灰度修正；不是整个原始 CT 无法测量，也不是所有去噪方法的精度上限。

|开发试件|实际基线 Dice|±1 时 Dice / 精确率 / 召回率的乐观上限|±1 时仍必错的假阳 / 漏判体素|参考孔匹配数上限，±1 → ±2|
|---|---:|---|---:|---:|
|2|0.92261|0.96060 / 0.94514 / 0.97656|4,227 / 1,748|71/73 → 73/73|
|3|0.96253|0.98247 / 0.98843 / 0.97659|17,089 / 34,989|22/27 → 24/27|
|5|0.97368|0.98759 / 0.98990 / 0.98530|9,390 / 13,734|11/12 → 11/12|

三件的 ±1 体素指标上限都高于门槛，但并不存在一个已证明能实现这些上限的算法。3 号的参考孔 **23、197、220** 在 ±2 时仍不可能达到 IoU 0.5。5 号的参考孔 **105** 有 137 个体素，其中 72 个即使在 ±2 内也不能判为孔；最大交集 65，IoU 至多 65/137，仍低于 0.5。这些反例解释了为什么要检查逐孔匹配而不只看 Dice。

### 如何复核这个上限

1. 在正式开发区 x=[360,616)、y=[376,632)、z=[400,528)，先只读原始图计算每个体素翻转 Bernsen 判定所需的最小整数扰动。半径在 1–8；编码绝对值 9 仅表示在 ±8 以内不能翻转。不同输出点可能要求同一输入字节作相反改动，因此把它们各自的最好结果放在一起是放宽约束，不是分割预测。
2. 3 × 8,388,608 个原始二值判定逐点与生产基线一致。对 639 / 640 / 640 个真实局部窗口另行枚举中心灰度和邻点极值，核对最小半径；单元测试还对小窗口逐个枚举各邻点灰度。三张签名半径图均在[读取此次评分参考前公开摘要](https://github.com/EtherealBian/cone-fdk/commit/c013c98)。见[预登记](BYTE-RADIUS-PILOT.md)、[原始封存记录](evidence/byte-radius/predictions.json)。
3. 逐孔导出每个扰动范围内不能变为孔的体素数。对大小为 N 的参考孔，若最多有 H 个体素能变为孔，任一合格预测对象至少有 64 个体素，故 IoU ≤ H / (N + max(0, 64 − H))。这是从已登记逐孔表作的**事后推导**，不假设预测对象能连通或不触边，不把“可能”计为实际匹配。
4. 第二个计数程序从 NIST 开发分割归档读取每一层，用真/假两张有符号灰度半径直方图重算全部 15 组体素界。对 112 个完整参考孔逐个扫描其包围框，复核每行及 ID/体素数，再用整数交叉相乘检验 IoU 和覆盖率。它不调用原评分函数。

[全部五档扰动的体素指标](evidence/byte-radius/results.json) · [逐孔匹配数上限与反例](evidence/byte-radius/coverage-bounds.json) · 逐孔原始表：[2](evidence/byte-radius/sample2-references.csv)、[3](evidence/byte-radius/sample3-references.csv)、[5](evidence/byte-radius/sample5-references.csv) · 签名图：[2](evidence/byte-radius/sample2.npz)、[3](evidence/byte-radius/sample3.npz)、[5](evidence/byte-radius/sample5.npz)。

```powershell
python scripts/ptx.py pores/search_support.cu build/pores/search-support.ptx
python scripts/ptx.py scripts/bernsen_radius.cu build/pores/bernsen-radius.ptx
python scripts/pore_byte_radius.py predict --cache $cache --out build/radius-predict-new
python scripts/pore_byte_radius.py score --cache $cache --pred build/radius-predict-new --out build/radius-score-new
python scripts/pore_radius_audit.py --cache $cache
```

上限不能当作正式验收成绩。未使用保留试件 4、6，不改变 [FINAL-GATE](FINAL-GATE.md)。
