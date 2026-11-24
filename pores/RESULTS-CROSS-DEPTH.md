# 同一套权重换到另一段深度，逐孔精度还能保持吗？

**不能。** 在开发试件 2、3、5 上，邻域树在新深度的匹配孔体积误差 P95 全都高于熟悉的迁移区，也全都高于 15%。我们没有换模型或移动裁剪框找更好的结果。新区域是同一试件的另一段，不是新的独立试件，也不替代正式开发区或保留验收。

|试件|新深度中心树：匹配/预测/参考；P95|新深度邻域树：匹配/预测/参考；P95|邻域树熟悉区 → 新深度：参考覆盖|新深度邻域树 Dice；通过项|
|---|---|---|---|---|
|[2](evidence/cross-depth/sample2.json)|48/50/54；19.82%|50/53/54；23.82%|45/49 → 50/54|.94078；9/11|
|[3](evidence/cross-depth/sample3.json)|15/19/19；30.64%|16/19/19；25.28%|22/24 → 16/19|.97279；5/11|
|[5](evidence/cross-depth/sample5.json)|7/9/8；29.83%|7/9/8；23.27%|10/12 → 7/8|.98210；5/11|

- **不混用分母。** 新深度两种模型共同匹配的 47 / 15 / 7 个孔，中心树 → 邻域树的同孔 P95 分别为 19.97% → 17.20%、30.64% → 25.30%、29.83% → 23.27%。2 号另外新增 3 个、丢 1 个；若看全部匹配孔，P95 反而由 19.82% 升至 23.82%。不同深度从不按孔 ID 配对。
- **不能只看 Dice。** 3 号 Dice 0.97279，但 19 个合格参考只匹配到 16 个，质心偏移 P95 为 1.308 体素，高于 1 体素。5 号 Dice 0.98210，但两个覆盖率分别仅为 7/9 和 7/8，且体积中位误差 10.00%。
- **实际复现的是推理和计量，不是精度。** 三件各取不重叠的 z=[608,704)，每段 96 × 256 × 256 体素。新读图路径和直接取第 14 个次序统计量的 27 点中值重建，先在旧区域的四层与原前端检查：三件合计 8,878,080 个特征数值、1,572,864 个二值判定全部一致。新区域的六张掩膜共 37,748,736 个分数，CUDA 与 sklearn 最大绝对差为 0；两路共用同一个去噪器，不能据此声称独立复现历史软件。掩膜先公开，再打开新区域标签，两次独立进程从源标签重新评分；12 张 CSV 逐字节相同，4 份 JSON 排除运行秒数后相同。

[事前登记](CROSS-DEPTH-REPLICATION.md) · [读取新深度参考前公开的六张掩膜](https://github.com/EtherealBian/cone-fdk/tree/95295763c1072cacd88829b5783c6ecd11ab8438/pores/evidence/cross-depth) · [全部数值、逐孔 CSV 及重放摘要](evidence/cross-depth) · [同模型原始迁移区结果](RESULTS-LOCAL-PROFILE.md) · [未变更的正式门槛](FINAL-GATE.md)。

```powershell
python scripts/pore_cross_depth.py --cache $cache --out build/depth-repredict
python scripts/pore_cross_depth_score.py --cache $cache --out build/depth-score-a
python scripts/pore_cross_depth_score.py --cache $cache --out build/depth-score-b
python scripts/pore_compare_score_runs.py --first build/depth-score-a --second build/depth-score-b --arms center profile --out build/depth-replay.json
```
