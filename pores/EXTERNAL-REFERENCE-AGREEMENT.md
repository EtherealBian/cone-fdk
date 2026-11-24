# 同一个孔，换一份分割能差多少？

在四件 AMBench 疲劳试件上，同一个孔以 Binary 与 UNet 两种公开分割计数，匹配孔体积差的 P95 为 **18.48%–39.39%**。也有一致的对照：RBF_105 上 Binary 与 Adaptive 的 P95 为 **12.22%**。这些数字衡量的是**两份数字分割之间的差异**，没有将任何一种指定为实物真值，也没有运行 ConeFDK 的灰度分割候选 E。正式开发结果仍见 [E 基线](RETAINED-BASELINE.md)。

## 逐件结果

每种方法均取发布分割体中央 256 层、横向两轴中央各一半边长（大小向下取整）；裁剪只用数组形状。孔用六连通标记，排除触及裁剪六面中任一面的结构和小于 64 体素的孔。两边都合格且 IoU ≥ 0.5 才做一对一匹配。下表每一对按左侧方法 A 作分母：`100 × |B − A| / A` 的第 95 百分位，未匹配的孔不进入该百分位。

| 试件 | 合格孔 Binary / Adaptive / UNet | Binary → Adaptive P95 | Binary → UNet P95 | Adaptive → UNet P95 | Binary 与 UNet 匹配数 | 其中 ±15% 区间不相交 |
|---|---:|---:|---:|---:|---:|---:|
| RBF_05 | 51 / 61 / 55 | 47.01% | 39.39% | 16.47% | 51 | 8 |
| RBF_09 | 57 / 56 / 46 | 22.79% | 29.54% | 18.72% | 46 | 9 |
| RBF_21 | 72 / 59 / 53 | 26.11% | 31.25% | 14.88% | 53 | 10 |
| RBF_105 | 26 / 31 / 18 | 12.22% | 18.48% | 16.52% | 18 | 1 |

168 个 Binary–UNet 匹配中，28 个孔的两个体积各放宽 ±15% 后仍无交集（`115 × min(A,B) < 85 × max(A,B)`）。对这样的**单孔**，不可能给出一个数同时落入两个区间。这不能推导出 E 的精度下限，更不能证明物理误差。方向也因试件而异：在 RBF_05 的 51 个匹配中，UNet 有 48 个较大、2 个较小、1 个相等；RBF_09 的 46 个则全都较小。

例如，以下为 Binary 首个体素在**发布分割体**（非原始扫描）上的零基 `(z,y,x)` 坐标。RBF_05 的 `(1143,693,406)` 在 Binary / UNet 中为 **66 / 96** 体素，交集 66；RBF_09 的 `(1341,741,550)` 为 **165 / 109**，交集 107。RBF_05 的 `(1057,690,701)` 两者都数出 **95** 体素，但交集只有 **88**：相同体积也不代表边界相同。

## 原档与复核

- 数据：Miner、Hobdari、Stanton、Narra 的 [CMU KiltHub AMBench 分割数据 v1](https://doi.org/10.1184/R1/31378690.v1)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本文的裁剪、连通体和配对统计为派生结果。按发布目录身份、字节数和 MD5 验证源文件；[逐孔记录](evidence/external-reference/volume-agreement.json)另存各源 SHA-256、裁剪范围、组件标签、种子位置、整型体素数、匹配交集以及未匹配的两侧标签。
- 两份阈值法数组以 `255` 表示孔；UNet 以 `1` 表示孔。用每件四张 `Test_Data/Primary` 人工截面对照，分别验证各前景编码在已标孔像素上的占比高于在金属像素上的占比；这四张不连续的截面不能提供整孔体积。作者 [README](https://ndownloader.figshare.com/files/62042347) 指定这些 z 是相对发布分割体的索引。未取得原始 NIST 灰度，未验证与未裁剪原图的配准。
- NPZ 三轴 spacing 均为 `5.5002e-6`，按米换算为 5.5002 µm，而 README 写 5.502 µm。原生表按前者换算，仅用来核对同一输入上的数值实现；上表全用体素，未据此宣称物理标定精度。
- 第一条路径从压缩数组流式读取裁剪；第二条路径用 NumPy 整卷解码后独立裁剪，交给原生 CUDA 程序按 5 和 31 层分块。两次每件每法的 CSV 逐字节对照，再用 SciPy 重建所有组件的 ID、种子、计数、触面、体积、等效直径和质心。配对另用项目原匹配器复算，分别核对两种分母的百分位。四件 × 三种方法共核对 3,170 条连通体记录（含小孔与触边结构，不能视作 3,170 个不同的物理孔）。见[原生重放摘要及 CSV 指纹](evidence/external-reference/native-replay.json)和[十二张原生逐孔表](evidence/external-reference/native-csv/)。以下命令在仓库根目录、已安装 NumPy/SciPy 和项目原生程序的环境中运行，输出位置必须未使用过：

```powershell
python scripts/pore_external_volume_agreement.py --cache build/amb-external-volumes --out build/amb-new-agreement.json
python scripts/pore_external_volume_replay.py --cache build/amb-external-volumes --evidence build/amb-new-agreement.json --out build/amb-new-native
Get-FileHash build/amb-new-agreement.json -Algorithm SHA256
```

本次逐孔记录 SHA-256：`c936b4d9292ebaed30a234a54fefecde5bdb8b838b1084a161c512aee42403a0`。同一数字掩膜可重复计数，和从灰度确定可信边界，是两件不同的事。后者需要同孔配准的灰度及可信连续三维标注；这四件不参加 [FINAL-GATE](FINAL-GATE.md)，不替代开发或保留试件。
