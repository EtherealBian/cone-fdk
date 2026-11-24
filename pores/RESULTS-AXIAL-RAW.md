# 相邻原始切片：找回被连到边界的孔，但没有稳定降低体积误差

用完全相同的训练补丁、起始权重与网络，只在 `axial` 臂开放 z±1、z±3 的四个原始灰度差异通道。`plane` 对照也保留原有 3D 中值滤波。三件开发试件在 **z=[256,352) 的迁移区** 上 Dice 都上升；2 号少匹配一个孔，3 号的配对孔体积误差反而上升。**不晋级**，不重跑正式区，不使用 4/6。

|开发试件|匹配 / 合格预测 / 合格参考：plane → axial|Dice|同孔数；同孔体积误差 P95|各自全部匹配孔 P95|研究区通过项 / 11|
|---|---|---|---|---|---|
|[2](evidence/axial-raw/sample2.json)|43/45/49 → 42/44/49|.94012 → .94860|42；25.00% → 17.43%|28.67% → 17.43%|7 → 7|
|[3](evidence/axial-raw/sample3.json)|20/22/24 → 19/21/24|.96500 → .97091|17；16.26% → 17.64%|18.78% → 20.90%|8 → 6|
|[5](evidence/axial-raw/sample5.json)|9/10/12 → 10/10/12|.97496 → .97872|9；21.17% → 22.45%|21.17% → 22.34%|6 → 6|

**5 号参考孔 15：恢复的是一次独立测量，不只是多找到三个相交体素。** 参考有 452 体素。plane 的 391 个相交体素都连入一个触及裁边的 1,852 体素对象，IoU 0.20439。axial 的最佳对象不再触边：400 体素，其中 394 个在参考内，IoU = 394/458 = 0.86026；比参考体积小 52 体素（11.50%）。另有两个参考体素仍连在触边对象里。原始坐标包围框 x=[486,499)、y=[584,595)、z=[258,269)。与[前一次不同模型的截面对照](RESULTS-SECTION-AREA.md)是同一个参考孔，不能混用两个对照臂的体积。[从源分割重新标号的局部相交账](evidence/axial-raw/trace-5-15.json)。

2 号失去参考 ID 12；3 号失去 71、108、196，同时找回 125、186。全部匹配、未匹配对象与 12 份原生 CSV 均在[证据目录](evidence/axial-raw)。逐臂以 8/37 层分块测量，对象表字节一致；2/3/5 的 plane→axial 表行数为 122→113、206→200、103→88，另由 SciPy 逐对象检查。每种方法的全部匹配孔与两种方法都匹配的孔是不同统计总体。

[预登记](AXIAL-RAW-PILOT.md) · [评分前公开的权重与预测摘要](https://github.com/EtherealBian/cone-fdk/commit/1fc1390b366b1b687e22dedef769d55e0171b723)。训练使用每件原有 24 层校准分割，未将迁移区标签用于拟合。两层一批的 Torch 对照因主机提交内存不足在完成 2 号后中断；保留两个权重。随后单层 CUDA 与独立 FP64 NumPy 卷积从原始档完整重放三件，2 号复现先前两个原生掩膜。共 6×6,291,456 个二值判定，后端差异为 0，最大绝对 logit 差 7.744×10⁻⁶。两路共用灰度预处理；不能将其解释为独立验证去噪器或实物计量精度。

```powershell
# $cache 指向已核验的开发源档；使用已锁定的研究依赖、原生程序和 PTX。
$e = 'pores/evidence/axial-raw'
python scripts/pore_axial_score.py --cache $cache --pred $e --out build/axial-rescore-new
python scripts/pore_context_trace.py --study axial-raw --specimen 5 --ref 15 --cache $cache --out build/trace15-new.json
# 从权重重推，勿将输出写回证据目录。
$d = 'build/axial-from-weights-new'; New-Item -Type Directory $d
Copy-Item "$e/fit.json","$e/plane.npz","$e/axial.npz" $d
foreach ($i in 2,3,5) { python scripts/pore_axial_replay.py --cache $cache --out $d --specimen $i }
python scripts/pore_axial_replay.py --cache $cache --out $d
```
