# 用一种方法找孔、用另一种方法量边界：只有 5 号的误差改善能重复核对并通过筛选

在同一段 96 层开发迁移区，保持邻域树找到的孔，只在两张掩膜都找到同一个完整对象时，把边界换成原 census 的边界。反向交换是预先指定的对照，不用来按试件择优。三件都没有丢掉正向检测器原有的任何参考匹配，但**覆盖不变不等于体积量得更准**。

|试件|正向匹配 / 合格预测 / 合格参考|同孔体积 P95：邻域树 → 正向|反向匹配 / 合格预测；P95：census → 反向|正向 Dice；研究区通过项|
|---|---|---|---|---|
|[2](evidence/reciprocal-object/sample2.json)|45 / 46 / 49|18.92% → 17.38%|42 / 44；14.89% → 20.39%|.95036；9/11|
|[3](evidence/reciprocal-object/sample3.json)|22 / 22 / 24|15.58% → 19.17%|20 / 22；19.36% → 15.78%|.97283；8/11|
|[5](evidence/reciprocal-object/sample5.json)|10 / 11 / 12|19.31% → 16.97%|9 / 11；17.01% → 19.82%|.98021；9/11|

在各方向与各自检测器的比较中，匹配参考 ID 完全相同，所以表内的同孔 P95 也是各自所有匹配孔的 P95。与另一个检测器的交集则不同：正向相对 census 分别多匹配 #56/63/93、#71/184、#62；共同 42、20、9 个孔的 P95 分别为 16.92%、19.36%、17.01%。不把不同分母当成同一项改善。

2 号下降 1.54 个百分点，不到预登记的 2 个百分点，Dice 还下降了 0.00227（最多允许 0.002）。3 号体积误差变大，只有负对照方向有改善。5 号达到预登记的改善幅度，但 16.97% 仍高于 15%，10/12 的参考覆盖也仍低于 90%。**三件合取筛选不通过**，本方案不升级正式候选。

三个试件分别有 32 / 15 / 8 个区域满足交换条件，实际改变 2,130 / 478 / 271 个体素。每区独立核验并集大小、原始分量数、裁边、体素交集与有理数 IoU。CPU 从原生 CSV 种子重新洪泛，与另一条连通分量选择及 CUDA 写回逐体素一致；从源标签独立重新评分，24 张 CSV 逐字节一致，4 份 JSON 只有原生运行秒数不同（[复核摘要](evidence/reciprocal-object/replay.json)）。所有分量均保留，不能只从评分 CSV 中挑出好例子。

[事前协议](RECIPROCAL-OBJECT-PILOT.md) · [评分前公开的六张掩膜和逐区账](https://github.com/EtherealBian/cone-fdk/tree/52b3b17dcd16070e897c1ced81dc14184b7a0794/pores/evidence/reciprocal-object) · [所有方向、逐孔表、未匹配参考 ID 与精确门槛](evidence/reciprocal-object) · [正式开发 H 的结果](RESULTS-H.md)。

```powershell
python scripts/pore_reciprocal.py --out build/reciprocal-repredict
python scripts/pore_reciprocal_score.py --cache $cache --out build/reciprocal-rescore
```
