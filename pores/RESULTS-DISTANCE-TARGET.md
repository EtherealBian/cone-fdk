# 给同一批边界位置标出距离，比只标孔和材料更准吗？

**距离监督未通过跨试件配对筛选。** 只在原校准区内训练：同一网络、同一初始化、同一批次和增强，一臂只给出界面哪一侧（±0.5），另一臂给出到最近相反标签中心的距离减半体素，再加符号。仅使用距裁剪面至少四格、四格内见过相反类别的中心。它是数字标签的距离代理，不是实物表面位置。

|试件|可监督中心：孔 / 材料|其中距相反中心 >1：孔 / 材料|含这两类较深中心的 z 层数（孔 / 材料）|
|---|---:|---:|---:|
|2|11,433 / 51,298|5,997 / 44,091|16 / 16|
|3|141,304 / 289,274|90,608 / 232,225|16 / 16|
|5|111,875 / 149,684|79,458 / 115,840|16 / 16|

|试件|算法|匹配 / 合格预测 / 合格参考|全部匹配孔 P95|Dice|迁移区通过项 / 11|
|---|---|---|---:|---:|---:|
|[2](evidence/distance-target/sample2.json)|原基线|42/44/49|14.89%|0.94702|9|
|[2](evidence/distance-target/sample2.json)|仅符号|39/43/49|30.73%|0.92787|5|
|[2](evidence/distance-target/sample2.json)|有符号距离|40/44/49|25.51%|0.93404|5|
|[3](evidence/distance-target/sample3.json)|原基线|20/22/24|19.36%|0.97291|6|
|[3](evidence/distance-target/sample3.json)|仅符号|14/18/24|28.29%|0.96259|4|
|[3](evidence/distance-target/sample3.json)|有符号距离|19/22/24|21.19%|0.96768|4|
|[5](evidence/distance-target/sample5.json)|原基线|9/11/12|17.01%|0.98011|8|
|[5](evidence/distance-target/sample5.json)|仅符号|9/10/12|40.71%|0.97244|5|
|[5](evidence/distance-target/sample5.json)|有符号距离|9/10/12|26.28%|0.97565|6|

|试件|对照 → 距离|共同孔数|共同孔体积误差 P95|新增 / 丢失参考 ID|未满足的登记条件|
|---|---|---:|---|---|---|
|2|原基线 → 距离|40|12.27% → 25.51%|无 / 22、76|P95 降至少 2 个百分点、Dice、参考孔覆盖、预测孔覆盖、全匹配 P95 不升|
|2|仅符号 → 距离|39|30.73% → 25.57%|80 / 无|无|
|3|原基线 → 距离|17|19.86% → 23.83%|139、184 / 108、186、196|P95 降至少 2 个百分点、Dice、参考孔覆盖、预测孔覆盖、全匹配 P95 不升|
|3|仅符号 → 距离|14|28.29% → 18.51%|40、104、137、151、195 / 无|无|
|5|原基线 → 距离|9|17.01% → 26.28%|无 / 无|P95 降至少 2 个百分点、Dice、全匹配 P95 不升|
|5|仅符号 → 距离|8|41.09% → 26.55%|22 / 15|无|

两个训练方法各更新 1,536 次，各累计监督 2,007,927 个带内中心（包含重复）；这不是独立样本量。共享 CUDA 原图特征后，原生卷积和 NumPy FP64 卷积逐体素比较，两种新方法各自的不一致数（仅符号 / 距离）：2 号 0 / 0；3 号 0 / 0；5 号 0 / 0。最大绝对输出差、非零分歧的首个坐标见预测清单。另一个进程从相同数据重新训练，两套权重及训练清单逐字节一致；按球内 256 个非零偏移显式枚举，三件全部 4,718,592 个校准距离目标与距离变换的字节摘要也相同（[独立复核](evidence/distance-target/explicit.json)）。三种方法的原生对象表在 8 / 37 层分块间逐字节复现，并由 SciPy 核对。

[事先登记的方法与否决条件](DISTANCE-TARGET-PILOT.md) · [校准覆盖](evidence/distance-target/census.json) · [开分前公开的两套权重、六张预测及代码](https://github.com/EtherealBian/cone-fdk/tree/7e5e52a6256bc3bfd548de65f7150575c54b1efc/pores/evidence/distance-target) · [逐孔表与完整比较](evidence/distance-target)。

```powershell
python scripts/pore_distance.py fit --feature-dir build/axial-raw-dev --cache $cache --out build/new-distance
python scripts/pore_distance.py predict --cache $cache --out build/new-distance
python scripts/pore_distance_score.py --cache $cache --out build/new-distance-score
```

以上只评已研究过的 z=[256,352) 迁移区。对照为 NIST 数字分割，不是对实物尺寸的认证。[正式验收](FINAL-GATE.md)分开报告，保留试件 4、6 未启用。
