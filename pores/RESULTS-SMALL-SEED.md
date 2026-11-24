# 补足小孔，能不多算吗？

**能找回孔，但这条规则会多算更多，不能晋级。** 对三个已公开的开发迁移区，取原 census 掩膜中所有完整的 8–63 体素种子，不修改任何原来就合格的预测。用每个种子自己的原图内外灰度，决定在离种子两体素以内向哪里长。2/3/5 各有 17/27/10 个种子。36 个种子长到 ≥64 体素；其中仅 3 个新匹配了合格参考孔，另外 33 个成了未匹配预测。后者不等于在材料中凭空制造了 33 个孔。

|试件|原有匹配 / 预测 / 参考 → 按灰度生长 → 等体积几何对照|匹配体积误差 P95，原 / 灰度 / 几何|Dice，同序|迁移区通过项（共 11）|
|---|---|---|---|---|
|[2](evidence/small-seed/sample2.json)|42/44/49 → 44/58/49 → 45/58/49|14.89 / 20.57 / 33.06%|.94702 / .94278 / .94313|9 / 7 / 8|
|[3](evidence/small-seed/sample3.json)|20/22/24 → 21/39/24 → 21/39/24|19.36 / 22.51 / 22.51%|.97291 / .97247 / .97249|6 / 5 / 5|
|[5](evidence/small-seed/sample5.json)|9/11/12 → 9/16/12 → 9/16/12|17.01 / 17.01 / 17.01%|.98011 / .97988 / .97991|8 / 8 / 8|

几何对照让每个种子增加**完全相同数量**的体素，优先取最近可达位置，不用灰度选方向。2 号的参考孔 87，63 体素原种子两臂均长到 140 体素，灰度臂 IoU 0.58219 略好于几何臂 0.57143；但孔 12 的灰度臂 0.57447 不及几何臂 0.66917。几何臂另外找回孔 111（IoU 0.50676）。3 号两臂都找回孔 71，5 号均未找回新孔。三个试件原来匹配的 ID 全保留且同孔 P95 不变。机制筛选五条有四条成立，**不新增未匹配预测**这一条在每件上均失败，不能只报找回的 3 个孔。

### 新增的未匹配预测，原位置有什么？

评分后重新从原始参考分割读取同一区域，以**最大非零交集**定位灰度臂的每个新增预测（不是改变匹配规则）。

|试件|新增未匹配预测|最大重叠是完整但不足 64 体素的参考|触边参考|合格参考但未配对|没有任何重叠|
|---|---:|---:|---:|---:|---:|
|2|12|7|0|3|2|
|3|16|12|3|0|1|
|5|5|4|0|0|1|

33 个里面有 **23 个**主要重叠的确是小孔，只是数字参考本身不到计量下限。把它们推过 64 会恶化预测覆盖率。不能据此推断真实材料中孔的物理体积。两臂全部新增预测、原生 ID、大小、最大重叠参考和 IoU 都在[逐对象追查](evidence/small-seed/overlap.json)。

[事前协议](https://github.com/EtherealBian/cone-fdk/blob/8473d732849db97e25d5ea1f9e239a15daca22c6/pores/SMALL-SEED-PILOT.md) · [评分前公开的 6 份掩膜和全部种子](https://github.com/EtherealBian/cone-fdk/tree/2512ad59f54e04932e2e6a27bc162e79eeacd587/pores/evidence/small-seed) · [研究筛选与完整对象表](evidence/small-seed)。CUDA 七点秩统计与 CPU 逐体素一致（共 18,874,368 个原生灰度值），原始 halo 保留；CPU 上独立的传播与队列遍历核对所有增长决策。等体积几何臂和后处理连通性在 CPU；原生 CUDA 计量以 8/37 层重放并用 SciPy 核对原生对象。此处没有声称整条研究流水线都运行在 GPU。

```powershell
python scripts/ptx.py scripts/pore_seed_median.cu build/pores/seed-median.ptx
python scripts/pore_small_seed.py --cache $cache --out build/seed-new
# 评分固定读取仓库封存掩膜，另写新目录；事后追查直接再读源档参考。
python scripts/pore_seed_score.py --cache $cache --out build/seed-score-new
python scripts/pore_seed_entrants.py --cache $cache --score build/seed-score-new --out build/seed-score-new/overlap.json
```

未打开保留试件。正式开发结果仍是 [H](RESULTS-H.md)；数字参考与物理溯源的界限、最终停止规则见 [FINAL-GATE](FINAL-GATE.md)。
