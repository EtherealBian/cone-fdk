# 覆盖全部校准体素，2 号误差达标，为何还不能晋级？

**同样的网络、步数和训练预算，只改变一半补丁的抽法。** 每件的 24 张校准片分成 1,536 个不重叠的 32×32 方块。一臂有放回地随机抽，另一臂打乱后每块看一次；另一半边界补丁及两臂使用的增广、初始权重全部相同。两臂都启用相邻原始切片。研究只用 2/3/5 的既有校准，不把迁移区的标注放入训练。

|开发试件|有放回臂看到的不重复网格块；含边界补丁的体素覆盖率|匹配 / 合格预测 / 合格参考：有放回 → 遍历|共同孔数；体积误差 P95|各自全部匹配孔 P95|研究区通过项 / 11|
|---|---|---|---|---|---|
|[2](evidence/census-train/sample2.json)|966；73.19%|42/44/49 → 42/44/49|42；21.04% → 14.89%|21.04% → 14.89%|8 → 9|
|[3](evidence/census-train/sample3.json)|995；77.75%|20/21/24 → 20/22/24|19；17.68% → 19.52%|17.80% → 19.36%|6 → 6|
|[5](evidence/census-train/sample5.json)|984；75.52%|10/12/12 → 9/11/12|9；16.72% → 17.01%|16.27% → 17.01%|7 → 8|

遍历臂三件都监督到 1,572,864 / 1,572,864 个不同体素。2 号保留完全相同的 42 个参考 ID，P95 下降确实来自同一批孔，不是因为丢掉难测对象；但只匹配了 42/49 个参考孔，Dice 为 0.94702，仍未通过这两条。3 号失去 ID 184、得到 ID 153。5 号 Dice 略升至 0.98011，却失去了 452 体素的参考孔 15：有放回组量为 407 体素（IoU 0.85529），遍历组有 406 个相交体素被连入触边的 1,902 体素对象，IoU 仅 0.20842。[原始分割重新标号的局部证据](evidence/census-train/trace-5-15.json)。

这是 z=[256,352) 的开发迁移区，不是正式区。只有 2 号通过预登记的配对改进要求；3、5 号失败，不更换为有放回臂、不给不同试件各挑一种方法。正式 H 的分数不变。

[预登记](CENSUS-TRAIN-PILOT.md) · [评分前公开权重、训练覆盖账与六张掩膜](https://github.com/EtherealBian/cone-fdk/commit/511856488bdf3689a52494b8e1f5f050950d0667) · [完整证据](evidence/census-train)。独立 FP64 卷积与原生 CUDA 对 37,748,736 个判定逐个一致（共享前端），最大绝对 logit 差 1.132×10⁻⁵。两种分块深度 8/37 的对象表逐字节相同；SciPy 检查表行数 2/3/5 分别为 114→118、228→193、98→99。

```powershell
$e = 'pores/evidence/census-train'
python scripts/pore_census_score.py --pred $e --cache $cache --out build/census-rescore-new
$d = 'build/census-from-weights-new'; New-Item -Type Directory $d
Copy-Item "$e/fit.json","$e/replacement.npz","$e/census.npz" $d
python scripts/pore_census_predict.py --out $d --cache $cache
# 重新拟合可使用 pore_axial_raw.py fit 生成的 cal-2/3/5.npy。
# 训练入口先按已公布的源档、特征和校准标签 SHA 校验，不能换成迁移区缓存。
python scripts/pore_census_train.py --feature-dir build/axial-raw-dev --cache $cache --out build/census-fit-new
```
