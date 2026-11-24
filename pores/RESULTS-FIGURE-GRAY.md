# 论文的去噪中间图来自原始扫描的哪一层？

**在 2 号的 900 张原始 TIFF 中定位到第 447 层（零基）。** 原图和中值图的两项配准对照通过。用同一几何和亮度映射比较去噪前后的灰度变化，固定去噪器比不去噪更接近论文插图；改带宽后，相对固定方法只改善 **1.08%**，未达到事先要求的 10%。不把新参数用于正式候选。

### 从总览定位，再用另一幅面板作对照

只用印刷总览中的原始灰度，在材料内部去除比例尺、标框和边缘，比较 900 层、8 个方向及水平和竖直各 ±8 格位移的高通归一化相关系数（NCC）。第 447 层 NCC **0.828003**，距它超过三层的最佳竞争者是第 451 层（0.644679），差 0.183324。放大的原图仅在左侧拟合平移、尺度和灰度线性映射。右侧 67,424 个显示像素的原图 NCC **0.987244**；固定同一映射，把原始三层做 3×3×3 中值后与论文的中值面板比较，NCC **0.987141**，RMSE 为该面板 P95−P5 灰度跨度的 **4.603%**（按原图跨度为 4.319%）。中值面板未参与配准。以上均是出版图的一致性检查，不是实物尺寸精度。

### 直接比较滤波带来的灰度变化

在接触论文第三张灰度面板的评分前，从 NIST 原始档封存了原图、中值和 12 组去噪输出：两种搜索半径、各六个带宽。下表比较“预测去噪减中值”与“出版去噪减中值”的 RMSE，单位是显示灰度，越小越好。在左区选定 21×21 搜索、5×5 补丁、h=40，右区不再选择。

|方法|左区 RMSE|右区 RMSE|右区弱梯度|右区强梯度|
|---|---:|---:|---:|---:|
|中值后不再去噪|5.0271|5.5673|4.7609|6.3050|
|固定 21×21、h=30|3.8542|4.4913|3.9880|4.9659|
|左区选定 21×21、h=40|3.8162|4.4430|3.9592|4.9005|

h=40 相对中值对照降低 20.19%，相对 h=30 降低 1.08%；必须两者都降低至少 10% 才能晋级。固定方法本身有可测的去噪效果，但这不能识别原 Avizo 的精确参数。灰度比较使用的原始邻域范围分别为 `[631,385,727,588)` 和 `[684,385,821,588)`（x0,y0,x1,y1），都在正式区的 x 上界 616 之外。左右显示点分离，但原始邻域部分重叠；67,424 个缩放像素不是独立样本。一张 JPEG 插图不能给出原生灰度真值，更不能代表另外两件试件的三维孔体积。未评分论文的二值面板，未访问保留试件。

[原论文，图 12](https://doi.org/10.1016/j.addma.2017.06.011) · [原始档，NIST](https://doi.org/10.18434/M32162) · [定位登记](https://github.com/EtherealBian/cone-fdk/blob/6f6cacf29859079d2ad9cea8ac63253bc9ff0754/pores/FIGURE-LOCATOR-PILOT.md) · [已公开的定位结果与灰度比较协议](https://github.com/EtherealBian/cone-fdk/tree/a66eb0ebe2097e2c31af2839a884f89c55a37dd3/pores) · [评分前的原生字段封存](https://github.com/EtherealBian/cone-fdk/tree/326afa7e43583b0d45ca777ac58a3ae3715732ce/pores/evidence/figure-gray) · [全部带宽与梯度分层](evidence/figure-gray/check.json) · [900 层定位分数](evidence/figure-locator/locator.json)。

**独立重算：** 从固定 SHA 的原论文重新提取灰度面板，核对四个裁剪文件及原生字段存档的 SHA。另一路径用显式双线性插值和独立相关、范数计算重算 420 个计数和指标，最大绝对差 2.04×10⁻¹⁴ 以内，左区选择、右区两项比值及晋级判断均一致。[复核记录](evidence/figure-gray/audit.json)。此项独立的是插值与统计，不是独立采集或第二种去噪实现。仓库不附论文及其像素；`fields.npz` 来自 NIST 原始档。

```powershell
# $pdfpy 指装有 pypdf 和 Pillow 的 Python；$py 为研究环境。
& $pdfpy scripts/pore_figure_panels.py --pdf $pdf --out build/new-panels --include-nlm
& $py scripts/pore_figure_locator.py --cache $cache --panels build/new-panels --out build/new-locator.json
& $py scripts/pore_figure_register.py --cache $cache --panels build/new-panels --locator build/new-locator.json --out build/new-registration.json
& $py scripts/ptx.py scripts/window_units.cu build/pores/window-units.ptx
& $py scripts/pore_figure_gray.py predict --cache $cache --out build/new-fields
& $py scripts/pore_figure_gray.py score --panels build/new-panels --pred build/new-fields --out build/new-gray.json
& $py scripts/pore_figure_audit.py --panels build/new-panels --out build/new-audit.json
```
