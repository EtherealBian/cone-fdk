# 相邻五层一起训练，哪些孔测得更准？

在 2 号开发迁移区，原模型与五层模型共同匹配到的 42 个孔，体积误差 P95 从 **14.89% 降到 10.96%**。这种改善没有跨试件复现。下表在同一原始扫描、同一段 96 层上对照：C 是已封存的 census 模型；S、V 从相同初始权重训练，分别输入单层、相邻五层的特征。S 和 V 的所有卷积权重均可训练。

|试件|匹配 / 合格预测 / 合格参考（C；S；V）|各自匹配孔体积误差 P95（C / S / V）|Dice（C / S / V）|迁移区通过项 / 11（C / S / V）|
|---|---|---|---|---|
|[2](evidence/five-plane/sample2.json)|42/44/49；46/47/49；44/45/49|14.89 / 21.19 / 11.09%|.94702 / .94159 / .94964|9 / 8 / 9|
|[3](evidence/five-plane/sample3.json)|20/22/24；18/22/24；20/26/24|19.36 / 17.15 / 23.28%|.97291 / .96737 / .97250|6 / 7 / 5|
|[5](evidence/five-plane/sample5.json)|9/11/12；9/10/12；9/10/12|17.01 / 13.85 / 21.89%|.98011 / .97669 / .98010|8 / 7 / 7|

|试件|C → V 同孔数：体积误差 P95|相对 C 新增 / 丢失参考 ID|S → V 同孔数：体积误差 P95|相对 S 新增 / 丢失参考 ID|
|---|---|---|---|---|
|2|42：14.89 → 10.96%|63、111 / 无|43：19.24 → 11.10%|111 / 12、56、87|
|3|18：17.19 → 24.02%|139、184 / 137、196|15：18.04 → 21.16%|80、125、139、184、186 / 71、137、196|
|5|9：17.01 → 21.89%|无 / 无|9：13.85 → 21.89%|无 / 无|

2 号 V 的参考覆盖率 44/49 = 89.7959%，Dice 为 0.949643，两项均不能取整过线。相对 S 的两侧覆盖率都降低。3 号 C 和 V 各匹配 20 个，但不是同一批孔；5 号则是在完全相同的九个孔上变差。没有一件同时通过两组预定筛选，**不将 V 晋级为正式候选**。

[训练前登记](https://github.com/EtherealBian/cone-fdk/blob/77faa30c3100f01a94ba1bf0db312ca666120bb6/pores/FIVE-PLANE-PILOT.md) · [评分前公开的两套权重及六份掩膜](https://github.com/EtherealBian/cone-fdk/tree/0a44708835668407b423c557e405e45a73fb10d7/pores/evidence/five-plane) · [全部对象、匹配 CSV 与十一项结果](evidence/five-plane)。在共享去噪前端之后，六份输出共 37,748,736 个体素上的 CUDA 与独立 FP64 解码决策零差异，最大连续值绝对差 1.103×10⁻⁵ 以下。两种原生分块深度 8/37 的对象表逐字节相同，另用 SciPy 按对象核对。实现一致不代表精度达标。

```powershell
# 锁定环境及既有 24 层校准特征见 CENSUS-TRAIN-PILOT.md 与 AXIAL-RAW-PILOT.md。
python scripts/pore_five_plane.py fit --cache $cache --feature-dir build/axial-raw-dev --out build/five-new
python scripts/pore_five_plane.py predict --cache $cache --out build/five-new
# 从仓库公开的权重摘要和六份封存掩膜评分；输出目录必须是新的。
python scripts/pore_five_score.py --cache $cache --out build/five-score-new
```

此处 z=[256,352)，不是 [H 的正式开发区](RESULTS-H.md)。没有打开保留 4、6；未改 [FINAL-GATE](FINAL-GATE.md)。
