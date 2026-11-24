# 达到更远的邻域后，同一批孔的体积测准了吗？

**没有达到预定改善幅度。** 在三个试件各自的 96 层开发迁移区，扩大视野真正改变了体素判定，却没有改变匹配到的任何一个参考孔 ID。匹配率没有降低，但最差一端的体积误差也没有稳定缩小。

| 试件 | 三种方法相同：匹配 / 合格预测 / 合格参考 | 同孔体积误差 P95：原模型 → 原尺度头 → 四倍尺度头 | Dice，同序 | 各臂通过项（迁移区） |
|---|---|---|---|---|
| [2](evidence/two-scale/sample2.json) | 42 / 44 / 49 | 14.89% → 15.95% → 15.55% | .94702 / .94701 / .94714 | 9 / 8 / 8 |
| [3](evidence/two-scale/sample3.json) | 20 / 22 / 24 | 19.36% → 18.88% → 19.31% | .97291 / .97299 / .97294 | 6 / 6 / 6 |
| [5](evidence/two-scale/sample5.json) | 9 / 11 / 12 | 17.01% → 16.74% → 17.01% | .98011 / .98002 / .97985 | 8 / 8 / 7 |

相对原模型，原尺度头 / 四倍尺度头分别改变了 374 / 168、2,617 / 3,812、1,335 / 1,733 个体素。两种头的可训练参数和 1,536 次更新相同；四倍尺度使用扫描坐标锚定的 4×4 非重叠块与块中心双线性插值。每件两种对照，均仅缺“共同孔 P95 至少降低 2 个百分点”这一条研究筛选条件。不能只取 3 号从 19.36% 到 19.31% 的改善而忽略其幅度和其他试件。

[预登记](https://github.com/EtherealBian/cone-fdk/blob/2804438/pores/TWO-SCALE-PILOT.md) · [评分前的权重、九个掩膜及签名](https://github.com/EtherealBian/cone-fdk/tree/62064c5/pores/evidence/two-scale) · [逐孔 CSV 和全部十一项](evidence/two-scale)。每个臂/试件 6,291,456 处 CUDA / FP64 CPU 决策相同（原基线前端共享）；新网格另在全部输出与 CPU 核对，并对校准和迁移各 96 个位置/尺度/试件直接求块均值和插值。8/37 层分块的对象表字节相同，按 ID 核对 SciPy。数值相符验证实现，不能使精度筛选通过。实验只检验冻结原模型上的浅校正头，不能排除从头训练三维模型。

```powershell
python scripts/ptx.py scripts/pore_raw_grid.cu build/pores/raw-grid.ptx
# 重训需前次验证过的 24 层校准特征；不重训时复制已封存的 fit.json 及两个权重即可重推。
python scripts/pore_two_scale.py fit --cache $cache --feature-dir build/axial-raw-dev --out build/scale-new
python scripts/pore_two_scale.py predict --cache $cache --out build/scale-new
python scripts/pore_two_scale_score.py --cache $cache --pred pores/evidence/two-scale --out build/scale-score-new
```

此处不重命名正式候选，4、6 未启用。正式测量仍见 [H](RESULTS-H.md) 和 [FINAL-GATE](FINAL-GATE.md)。
