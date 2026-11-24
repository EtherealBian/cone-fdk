# 给“该连”和“该断”的体素加权，能把孔单独量出来吗？

**5 号的九个共同孔量得更准了，但三个试件均未通过配对筛选。** 新方法只改变训练损失，不在推理时用参考掩膜修补。它在 2 号多匹配了三个参考孔，在 3 号却少匹配了两个。下面的分母是各方法实际产生的合格预测与同一批合格参考；不是只统计成功的对象。

|试件|匹配 / 合格预测 / 合格参考：遍历对照 → 连通监督|共同孔数；体积绝对相对误差 P95|各自全部匹配的 P95|Dice|11 项中达到数|
|---|---|---|---|---|---|
|[2](evidence/local-connectivity/sample2.json)|42/44/49 → 45/48/49|40；12.53% → 18.90%|14.89% → 20.66%|.94702 → .93211|9 → 7|
|[3](evidence/local-connectivity/sample3.json)|20/22/24 → 18/26/24|17；18.30% → 45.92%|19.36% → 45.90%|.97291 → .96478|6 → 5|
|[5](evidence/local-connectivity/sample5.json)|9/11/12 → 9/14/12|9；17.01% → 12.42%|17.01% → 12.42%|.98011 → .97578|8 → 7|

2 号新增参考 ID 12、56、63、87、111，丢失 26、27；3 号新增 71，丢失 125、153、186；5 号匹配的参考 ID 完全相同，却多出三个未配预测。具体到 5 号的 **452 体素参考孔 15**：旧方法认出其中 406 个，但并入触边的 1,902 体素对象；新方法认出 411 个，却并入触边的 2,001 体素对象。两次都没有独立匹配。[所有相交分量、包围盒与交并比](evidence/local-connectivity/trace-5-15.json)。

训练只用了 2/3/5 各 24 张原校准切片。删除 3×3×3 邻域中心后，若至少两组孔体素各自六连通并接触中心的面，则按原中心标签分成应留间隙和应留连接两类。这是局部判据。三件不同间隙/连接中心分别为 **42/86、119/220、88/110**；训练重复见到它们的次数分别为 294/577、396/1154、185/374。两臂抽样、初始化与训练步数相同。没有针对上面这个孔修改临界点定义。

[预登记与判定规则](LOCAL-CONNECTIVITY-PILOT.md) · [校准清点](evidence/local-connectivity/census.json) · [评分前的权重、六张掩膜和摘要](https://github.com/EtherealBian/cone-fdk/commit/89dc028b702b3635ff5c1bf4c0077a0989c66cae) · [原生对象表、逐孔配对及所有失败项](evidence/local-connectivity)。原生 CUDA 与独立 FP64 卷积的 37,748,736 个二值判定完全相同；灰度前端共用。分块深度 8 和 37 的对象表相同，SciPy 逐行核对分别为 118→146、193→364、99→146 行。控制掩膜逐体素复现此前遍历对照。

```powershell
$e = 'pores/evidence/local-connectivity'
python scripts/pore_local_score.py --pred $e --cache $cache --out build/local-rescore-new
$d = 'build/local-weight-replay-new'; New-Item -Type Directory $d
Copy-Item "$e/fit.json","$e/census.npz","$e/critical.npz" $d
python scripts/pore_local_predict.py --out $d --cache $cache
python scripts/pore_local_train.py --feature-dir build/axial-raw-dev --cache $cache --out build/local-fit-new
python scripts/pore_context_trace.py --study local-connectivity --specimen 5 --ref 15 --cache $cache --out build/local-ref15-new.json
```

这些是 **z=[256,352) 的开发迁移区**，不是正式开发区、保留试件或实物孔径认证。正式 H 仍为 [7/11、4/11、5/11](RESULTS-H.md)。
