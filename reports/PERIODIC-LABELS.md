# 从一对壁厚阈值，到可带走的逐格标签

分带摘要能回答“有几片”和“两点是否相连”，却不能指出每一格属于哪一片。新的可选 CUDA 程序为选定的**全局阈值对**输出整张区域 ID 图。一位工程师可以在带 RTX 的机器上生成，另一位只拿结果包，在没有 CUDA 的电脑上独立检查标签。原来的重建和批检默认行为不变。

## 为什么复核不是简单比文件大小？

1. **生成时查来源。** 从外部提供的边缘作业清单摘要开始，现有分带路径先复算所有阈值组合、覆盖及保存的上下界，再从有摘要的两份原始壁厚重读所选组合。缺见证或滑移拒测的作业没有标签。一次只生成一对；不把三态掩码当作原始 CT 输入。
2. **GPU 标记。** 输入每格 1 字节：`0` 低于门槛、`1` 实测超限、`2` 不可测。对超限格做四邻域并查集，只有方位方向首尾相连。每片 ID 等于自身最小的行主序格号加一，未知格为 `−1`，不会填成“正常”。一张图用同一组阈值；九种组合不是九次独立观察。
3. **离线再证明。** 从可信渠道取得包清单的 SHA-256。离线端对三态掩码独立重数连通片、最大格数和两锚点状态，然后检查每一格。相邻超限格必须同号；不同正 ID 的数目必须等于独立片数；每个 ID 都要指回本片最早的一格。三项合起来既能识别把一片拆开，也能识别将两个不相连的岛误并。另核对所有未知格、低于门槛格和文件摘要。离线端不需要原投影；要复核壁厚本身，仍需上游作业。

## 在仓库里复现

使用仓库锁定的 Windows x64 Python 环境。独立构建不改原 EXE/PTX；下例使用 x64 MinGW。指定新的输出目录，两次编译的二进制均需一致。原生后端每次处理完整的一张图，只将同一批中的图顺次放入显存；若预算小到一张也放不下则拒绝，不会自动缩小输入。

```powershell
python scripts/build_periodic_labels.py build/periodic --cxx C:/msys64/mingw64/bin/g++.exe
```

`periodic-labels.exe` 是对已分类掩码的底层工具，只产生小端 int32，不验证 CT 来源。实际作业走 `scripts/periodic_parcel.py generate`，需要受信资格文件摘要和源作业清单摘要。`verify` 使用 NumPy/SciPy，不启动显卡或重建程序。包只包含 `manifest.json`、`mask.u8`、`labels.i32`；不会覆盖已存在的交付目录。不要将自己在待检查文件上刚算出的摘要当作信任根。

## 计时与边界

[在运行实验前冻结的规程](../fixtures/cuda-periodic-label-protocol-v1.json)区分两种时间：**新进程读入已分类掩码、生成并刷盘发布完整文件、直至退出**，以及两端各自的**标签计算阶段**。CPU 使用仓库已有的 SciPy/NumPy 实现，因此进程数值包含 Python 科学库启动成本；GPU 进程数值包含 CUDA 初始化与传输，CUDA 事件数值则不含传输。未锁时钟、未清文件缓存。既不是与最优原生 CPU 实现的对照，也不是从投影到检测的耗时。不据此推断实物精度。

### RTX 5060 Laptop 实验记录

| 口径 | CPU（SciPy/NumPy） | CUDA | 判据 |
|---|---:|---:|---|
| 9 张 1,536 × 960 掩码，进程从启动到完整输出退出 | 1,092.85 ms | 207.63 ms | 六次交替配对全部更快；两组中位数之比 0.190，输出字节相同 |
| 标签计算阶段 | 279.55 ms | 1.45 ms（CUDA events） | 不含进程启动或文件 I/O；不用于推算整条 CT 管线耗时 |

两次构建的 EXE、PTX 字节一致。穷举全部 729 张 2 × 3 三态图；另与 40 张历史原生图的 590,400 个标签逐一对照。大图的 13,271,040 个标签与独立 CPU 对照，三次设备输出摘要相同；程序自有设备缓冲（含护栏）峰值为 **13,272,576 B**，不是整卡显存占用。离线证书端除了行带，还需每个可能的 ID 一字节。已有目标、末图非法字节、截断和预算不足四类反例均拒绝发布。[原始计时、驱动和全部 SHA-256](periodic-label-qualification.json)。

从已验收的混合配方生成 **67 个包**，共 988,920 个配对格：八条模拟比较作业有图，两条上游拒测无图。每个包的标签都对上此前独立保存的原生摘要。不完整对照的原始摘要来自逐档后端，配方却采用合包后端；只在[既有等价性证据](packed-edge-qualification.json)的两个来源、同一父清单及全部八个产物字节都核对通过时，才采用旧标签作为对照。[逐包结果](periodic-parcels.json) · [完整回归](periodic-regression.json)。

### 只下载示例，在 CPU 上复核

在仓库根目录执行；包清单的受信摘要来自上面的逐包报告。无需编译、显卡或原投影。两个示例分别回答“跨缝有实测连接吗？”和“盲格能否改变结论？”。

```powershell
Expand-Archive reports/periodic-seam-demo.zip build/example-seam
python scripts/periodic_parcel.py verify build/example-seam --manifest-pin 961648e72ce25d1e3d36b1f8a576af3e63750a2b809ca4e160a183e075ba2395
Expand-Archive reports/periodic-blind-demo.zip build/example-blind
python scripts/periodic_parcel.py verify build/example-blind --manifest-pin 90cb28cac64a07fde8e5a64bc721adc5762e44fe3ab10b71bdc896967fa1e649
```

前者两锚点属于同一实测连通片；后者在 14,760 格中有 14,723 格不可测，两点可能经盲区连接，**不能报为已连或已断**。

### 在 GPU 开发机重做资格试验

先按[混合批检复现说明](STRIPED-BATCH.md)安装锁定的环境、构建原生 CT 程序，再按上文构建独立标签工具。以下目录均须是新输出。首次命令会重建模拟作业；资格重跑产生的新文件含本机计时，因此另外计算摘要，不覆盖仓库证据。

```powershell
python scripts/replay_stripe_mixed.py reports fixtures/stripe-campaign-plan.json build/my-replay.json --keep build/my-sources
python scripts/qualify_periodic_labels.py build/periodic reports build/my-sources/build/bridge-campaign/r1/measurement build/my-sources/build/regions-campaign/r1 build/my-label-study
python scripts/qualify_periodic_parcels.py build/my-sources/fixtures/stripe-campaign-plan.json reports build/periodic build/my-label-study/qualification.json build/my-parcels
```

实测核桃在完整回归中只验证重建；上述标签与连接结论来自模拟管件、已分类压力图和解析反例。没有新的实物尺寸真值。
