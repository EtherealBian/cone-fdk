# 13 个方向的三维灰度曲率：找回三个孔，体积尾部误差仍不合格

在相同的原始 CT、校准位置、训练行和判定阈值上，只给原 46 列模型增加 15 列灰度特征（3 个尺度、每尺度 5 个方向秩）。三件熟悉开发区都比原模型多匹配一个参考孔；原来匹配到的全部保留。但即使只比较原有同一批孔，体积误差 P95 也**全部上升**。不升级正式候选，不开 4/6。

|试件|匹配 / 预测 / 参考：原模型 → 新模型|全部匹配体积绝对相对误差 P95|共同孔数；共同孔 P95|新模型 Dice|研究区通过项|
|---|---|---|---|---:|---:|
|2|45/46/49 → 46/47/49|18.92% → 22.55%|45；18.92% → 21.08%|0.95372|10/11|
|3|22/22/24 → 23/23/24|15.58% → 18.36%|22；15.58% → 18.56%|0.97369|8/11|
|5|10/11/12 → 11/12/12|19.31% → 19.64%|10；19.31% → 20.10%|0.98086|10/11|

每件新模型的预测侧和参考侧覆盖都达到 90%。2 号新增 #87（参考 91、预测 64 体素，IoU 0.632）：达到计量下限，体积仍低估 29.67%。3 号新增 #139（148 → 167，IoU 0.831）；5 号新增 #15（452 → 426，IoU 0.876）。2 和 5 号只差体积 P95，3 号还差体积和等效直径的中位误差。这个比较使用 z=[256,352)；不是正式 z=[400,528) 或未见试件上的结果。改善的是找孔能力，不是测量精度。

## 怎么重算

[事前协议](VOLUME-CURVATURE-PILOT.md) · [评分前封存的模型与掩膜](https://github.com/EtherealBian/cone-fdk/tree/ef2e596ac712f4008888866a6e638b6744a1fb86/pores/evidence/volume-curvature) · [逐孔 CSV、全部指标与重放哈希](evidence/volume-curvature) · [正式验收线](FINAL-GATE.md)。每件新增 94,371,840 个整数特征逐个对照 CPU；所有六组模型/试件的 CPU/CUDA 决策与分数最大差均为零。两个目录独立重训及预测的十个文件逐字节相同；两次从标签源档评分，12 份 CSV 一致，4 份 JSON 仅排除原生计时后相同。原生计量在 slab=8 与 37 的输出及独立 SciPy 标签也做了核对。

在已准备依赖、原生程序、父实验 PTX 及校准特征缓存的仓库根目录运行（每次使用新的输出目录；评分器校验并读取已封存的预测，不隐式采纳刚跑出的模型）：

```powershell
python scripts/ptx.py pores/volume_curvature.cu build/pores/volume-curvature.ptx
python scripts/pore_volume_curvature.py fit --cache $cache --feature-dir build/axial-raw-dev --out build/curve-new
python scripts/pore_volume_curvature.py predict --cache $cache --out build/curve-new
python scripts/pore_volume_curvature_score.py --cache $cache --out build/curve-check
```
