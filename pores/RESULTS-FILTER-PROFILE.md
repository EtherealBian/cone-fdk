# 对照论文的两步灰度变化，误差从哪里来？

在已定位的试件 2 第 447 层，分别检查 **原图 → 中值** 和 **中值 → 非局部均值去噪**。以下 RMSE 用出版图的灰度级表示，比的是两图的变化，不是三维物理量。两项研究都锁定原图配准与灰度映射，用左区选参数、右区比较；这张图已经用于开发，右区并非新的盲样。

**第一步：保留原来的 27 点中值。** 在右区 67,424 个图像像素上，假设不做中值处理，变化误差为 **5.62228**；按原始 16 位体数据计算 3×3×3 中值，降至 **4.90155**（弱/强梯度区 4.45075/5.33467）。这是渲染流程的正对照，并不规定下游去噪误差能达到的下限。

|中值方法|左区变化 RMSE|右区变化 RMSE|
|---|---:|---:|
|原图，不做中值|5.39243|5.62228|
|完整 27 点（对照）|4.63902|4.90155|
|面邻域加中心，7 点|4.92298|5.16339|
|面和棱加中心，19 点|4.68226|4.93063|
|切片内正方形，9 点|4.84039|5.07747|

另有六种轴次序，每次做一维三点中值，右区 RMSE 为 5.02845–5.04760。九个替代中左区选出 19 点，右区变化误差上升 0.59%，绝对灰度 RMSE 从 3.49812 上升到 3.57147。十种中值各在 256 个位置用独立标量排序复算，原生 uint16 没有差异。这个结果支持在这张图上保留 27 点，不能唯一鉴定各件历史软件的全部设置。[协议](MEDIAN-SOURCE-PILOT.md) · [评分前原生字段](https://github.com/EtherealBian/cone-fdk/commit/23e35084b8b4b4350e0af5b7d6da3a5e683cdafb) · [全部方法和梯度分层](evidence/median-source/check.json)。

**第二步：改变 5×5 补丁内各位置的权重也没有改善结果。** 同样的 21×21 搜索、相同中心投票，只把补丁距离的均匀系数改成二项式系数。两臂左区均选 h=40；右区变化 RMSE **4.44302 → 4.47553**，没有达到降低 10% 的要求。固定 h=30 为 4.49132 → 4.51921。六个原生均匀字段复现；两种 CUDA 实现在三个非恒定位置、六个带宽上与显式双精度求和比较，最大差小于 0.0018 原生灰度级。

|开发试件|两臂在校准拟合区各自选出的 h|分离的 8 层校准检查区 Dice：均匀 → 二项式|假阳数：前 → 后|漏检数：前 → 后|
|---|---:|---|---|---|
|2|30 / 30|.94405 → .92542|250 → 372|89 → 89|
|3|30 / 30|.95829 → .94348|886 → 1339|1641 → 2086|
|5|35 / 35|.98131 → .97578|429 → 668|689 → 784|

2 号漏检总数相同，但修复 44 个旧漏检的同时又产生了 44 个新漏检。三件均未晋级，未将替代中值或二项式系数用于正式开发或保留试件。[协议](PATCH-PROFILE-PILOT.md) · [开分前字段、掩膜与全部 36 个拟合分数](https://github.com/EtherealBian/cone-fdk/commit/7156fa80b5e8b1b9f1118354075b43c1ab5c6e15) · [分区明细与八格转移账](evidence/patch-profile/check.json)。

```powershell
python scripts/ptx.py scripts/patch_profile.cu build/pores/patch-profile.ptx
python scripts/pore_patch_profile.py --cache $cache --out build/patch-profile-new
python scripts/pore_patch_profile_score.py --pred build/patch-profile-new --cache $cache --panels build/figure12-audit --out build/patch-check-new.json
python scripts/pore_median_source.py predict --cache $cache --out build/median-source-new
python scripts/pore_median_source.py score --pred build/median-source-new --panels build/figure12-audit --out build/median-check-new.json
```

图像裁片须从[先前登记的论文版本](RESULTS-FIGURE-GRAY.md)本地提取并核对 SHA。仓库只提供自己的数值字段、统计和摘要，不分发出版图。
