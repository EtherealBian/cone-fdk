# 换一种方法直接从灰度找孔：找回了 5 号的 #62，但尾部误差仍未达标

原来十二项研究都没匹配的三个孔中，这次在 **5 号**找回一个。算法不知道孔的 ID 或位置：用原有 24 层校准区拟合两组固定梯度提升树，在已经公开的开发迁移区直接生成掩膜。中心模型只看当前位置的十个特征；邻域模型另外看原始灰度、中值、去噪和阈值余量在三个半径上的次序统计。不是在已知掩膜里逐孔挑最好的一个。

|开发迁移区|原 census：匹配/预测/参考；体积 P95|中心树|邻域树|邻域 Dice；通过项|
|---|---|---|---|---|
|[2](evidence/local-profile/sample2.json)|42/44/49；14.89%|44/46/49；21.16%|45/46/49；18.92%|.95264；10/11|
|[3](evidence/local-profile/sample3.json)|20/22/24；19.36%|20/21/24；19.63%|22/22/24；15.58%|.97282；8/11|
|[5](evidence/local-profile/sample5.json)|9/11/12；17.01%|10/10/12；16.98%|10/11/12；19.31%|.98018；9/11|

- **2 号：**相对 census 新增 #56、#63、#93，旧匹配全部留下，达到双侧 90% 覆盖。迁移区只剩体积 P95 未过，但相同 42 个孔的 P95 由 14.89% 升到 17.46%，不能以增加覆盖掩盖这一退步。
- **3 号：**相对 census 新增 #71、#184，不丢旧孔。共同比较的 20 个孔 P95 从 19.36% 降至 14.13%；两种对照的机制筛选均通过。算上新增孔，全部 22 个的 P95 为 **15.58%**，不能按 15% 达标；体积中位误差与等效直径中位误差也未过。曾连到裁边的 #70 仍漏配。
- **5 号：**#62 原为 100 体素，中心树测为 114（IoU .540），邻域树为 112（IoU .582）。**仅中心特征已能发现它**，不能单独归功于邻域统计。对 census 共同的 9 个孔，P95 从 17.01% 升为 19.82%；仍缺至少一个匹配才能达到参考覆盖线。

每件拟合 393,216 个位置，各件校准孔体素分别为 4,905、64,228、63,288。两模型各 96 棵树，数值节点无需 pickle；两次从已核验校准特征和源标签重训，模型文件及 fit.json 逐字节相同。独立 CUDA 遍历与 sklearn 库的双精度树计算，在 3 × 2 × 6,291,456 个输出分数上最大绝对差为零。两条路径共用灰度前端，这不是端到端独立验证。每件每臂结果另经 8/37 层原生计量和 SciPy 连通体核验。

[事前登记](LOCAL-PROFILE-PILOT.md) · [打开迁移标签前公开的权重和六张掩膜](https://github.com/EtherealBian/cone-fdk/tree/0883870519a26450f9338725876733534daec8b2/pores/evidence/local-profile) · [全部逐孔 CSV 与新增/丢失 ID](evidence/local-profile) · [原先三个难例](RESULTS-PERSISTENT-MISS.md) · [正式 H 与未改变的验收门槛](RESULTS-H.md)。本方法仅 3 号通过研究筛选，不进入正式验收。

```powershell
python scripts/ptx.py pores/local_profile.cu build/pores/local-profile.ptx
# 复用 axial-raw 的原始校准特征，先核源标签和特征 SHA。若未生成见 RESULTS-AXIAL-RAW.md。
python scripts/pore_local_profile.py fit --cache $cache --feature-dir build/axial-raw-dev --out build/profile-new
python scripts/pore_local_profile.py predict --cache $cache --out build/profile-new
python scripts/pore_profile_score.py --cache $cache --out build/profile-score-new
```
