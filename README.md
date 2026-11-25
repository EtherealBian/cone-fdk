<a name="zh-cn"></a>

# ConeFDK · 单 GPU 锥束 CT 重建

**简体中文** · [English](#english)

在 NVIDIA DGX Spark（GB10）上，用 720 幅核桃 X 射线实测投影重建 **512³ 体数据**。原生 C++/CUDA 引擎完成锥束加权、cuFFT 分批滤波和按显存预算分块的反投影。Python 用来准备输入和独立校验；重建程序本身不需要 Python。

[个人主页项目详解](https://etherealbian.github.io/work/cone-fdk.html) · [Windows 稳定版 v0.1.0](https://github.com/EtherealBian/cone-fdk/releases/tag/v0.1.0) · [下载离线切片浏览器](https://github.com/EtherealBian/cone-fdk/releases/download/v0.1.0/walnut-512-viewer.html)

![720 角真实投影重建的核桃轴向切层](reports/walnut-sweep.gif)

动图中的壳、隔膜和核仁均来自实测数据：[Alexander Meaney / 赫尔辛基大学](https://doi.org/10.5281/zenodo.6986012)，CC BY 4.0。离线浏览器可拖动查看 81 张预览切片，无需 CUDA。[数据署名与处理说明](NOTICE.md)

## DGX Spark 实测：一颗核桃，三种重建网格

| 检查 | 实测结果 | 记录 |
|---|---|---|
| 720 角、512³ 重建 | 两次原生输出逐字节相同；与 Spark 上独立运行的 ASTRA FDK 相比，全体积相对 L2 差异为 **0.02899%**，未配准或拟合强度。 | [原生重复](reports/gb10/walnut-native.json) · [ASTRA 对照](reports/gb10/walnut-astra.json) |
| 一次重建三个网格 | 整件概览加两个不同网格的局部，从读三遍投影减为一遍。六组配对中，原生进程中位耗时 **1.743 → 0.643 秒**，三个输出与分别重建逐字节相同。 | [逐次计时与口径](reports/gb10/multigrid-timing.json) |
| 断点恢复 | 进程在第 2 块写入后被强制终止。沿用原有回调补算其余 3 块，512³ 输出与整卷基线逐字节一致。 | [中断与复核记录](reports/gb10/slab-recovery.json) |
| 无 GPU 复核 | 随仓库提供另外两个实测选区的原生与 ASTRA 体素。仅用标准 Python 即可复算 **138,948 个体素**的逐层差异。 | [运行步骤](platform/spark/QUICKSTART.md) · [样例数据](examples/gb10-walnut) |
| 整颗核桃哪里相差最大？ | 512 层中，差异最大的位置首次出现在 **(367, 318, 256)**。下载两种算法在该层的原始切片，即可逐一复算 262,144 对体素。 | [逐层曲线与运行步骤](platform/spark/QUICKSTART.md#offline-physical-peak) · [批检导出记录](reports/gb10/physical-peak-snapshot.json) |

**两件实物、三种扫描条件已完成批检。** 在 256³ 网格上，松果的原生 CUDA 与 ASTRA 相对 L2 差异为：90 幅投影 **0.07314%**，720 幅投影 **0.02658%**。每组直接重建的两个局部都与整卷对应区域逐字节一致。两个角度密度来自同一颗松果；这些差异衡量求解器一致性，不是实物尺寸公差。[批检结果](reports/gb10/physical-batch-complete.json) · [离线复算三个峰值层](platform/spark/QUICKSTART.md#completed-physical-batch)。下文未注明 Spark 的旧基准，耗时均来自各自记录的历史平台。

**Spark 迁移验收：** Spark 专用测试 929 项全部通过。原有 707 项中，702 项通过，剩余 5 项要求禁止其他进程改写或替换原文件名。[测量工具构建步骤、实测结果与兼容边界](platform/spark/QUICKSTART.md#spark-migration-acceptance)。

## 换一组阈值，缺陷区域会变多少？

在带局部壁厚变化的**仿真试件**上，参考图与待检图各取三个边缘阈值。九种组合都得到一个连通区域，但覆盖范围是 **253 至 275 个网格单元**；同一试件的重复对照在九种组合下均为零。另一个无效边缘对照保留原定的四种组合，无法判定的单元单独计数。Spark 上两条原生 CUDA 路径分别重算，与封存的壁厚及分类文件逐字节相同。[不用 GPU 复算 22 张连通区域图](platform/spark/QUICKSTART.md#offline-threshold-review) · [逐组结果](reports/gb10/directory-edge-qualification.json)。

## 不生成整件，直接重建选区

`--roi` 从原投影直接重建一个长方体，保留整件的坐标和体素间距。核桃的 128 × 128 × 96 选区只占 **6 MiB**，六次都与 512 MiB 整卷的对应裁剪逐字节一致。[选区示例与命令](reports/ROI.md)

多个选区可用 [`--roi-list`](reports/MULTI-ROI.md) 共用滤波后的投影。在历史 Windows 基准中，六个选区共用 48 MiB 预算时，[`--planner min-passes`](reports/ROI-PLANNER.md) 把投影读取从三遍减为一遍，进程耗时中位数从 0.590 秒降到 0.304 秒。以上选项需从源码构建。

## 沿指定方向看截面

`--plane` 直接从投影计算一个指定方向的切面。核桃的 320 × 288 斜切面仅占 **360 KiB**，无须生成整卷。[切面示例与 FP64 对照](reports/DIRECT-PLANES.md)

要沿同一方向连续查看结构，可用 `--stack` 输出带空间坐标的 NRRD。核桃的 21 层斜切片只读一遍投影，每层都与单独运行的结果逐字节一致。[斜切动图与命令](reports/OBLIQUE-STACK.md)

## 把圆周摊开看

平面切片只能看到环形结构的局部。`--cylinder` 可以沿指定轴把完整一圈铺成图，核桃的 720 个实测投影只读一遍就得到半径 7–17 mm 的 11 张展开图。[下载离线圆柱展开台](reports/walnut-cylinder-atlas.html)：拖动半径看内外层，移动接缝追踪结构，或点击两点量同一圆柱上的最短表面距离。核桃不是规则圆柱，图层也不等同于核桃壳边界。[截图、误差和复现命令](reports/CYLINDER-ATLAS.md)。

![实测核桃从半径 7 到 17 mm 的同轴展开图](reports/cylinder-sweep.gif)

## 找出管件哪一侧最薄

`cone-fdk wall` 沿每个高度和方位的半径找出内外界面，以毫米记录两者之间的距离。两件已知尺寸的模拟偏心管，最薄的 15° 扇区分别落在设计的 **358° 和 134°**，逐点壁厚最大误差分别为 **0.01322 mm、0.01065 mm**。给第一件加上与外管隔开的内衬后，6,120 条径向剖面全部识别为多层并拒测，不会把空气间隙算进壁厚。[看误差曲线、复核一条剖面或运行自己的扫描](reports/RADIAL-WALL.md)。这是当前源码的模拟验收，尚未用带尺寸真值的实物管件校准。

## 换个装夹位置，内外径还测得一样吗？

`cone-fdk bore` 在每个高度分别拟合内外圆，给出内外径、两个圆心的扫描坐标、偏心量和圆模型最小壁厚。对同一根管做模拟平移、旋转并重新生成投影后，原采样的外径最多变化 **0.01696 mm**；探测器间距和径向重建步长均减半时，变化降到 **0.00788 mm**，通过预设的 **0.012 mm** 门槛。单独加密重建不能让原扫描通过这项目标。[四组对照、椭圆拒测和复现命令](reports/BORE-GAUGE.md)。这是模拟结果，不是实物公差。

## 管件装斜了，能不能继续测内外径？

先用 `axis` 从外壁找出管轴，再从**同一份原始投影**重建垂直于它的截面。在 8° 和 12° 的模拟装夹中，原先每份扫描的 25 层都因斜切而拒测；找正后，25 层全部通过同一圆度门槛。计入一组带计数噪声的扫描，四组圆管的最大直径误差为 **0.0010905 mm**。外壁椭圆时不发布轴，只有内孔椭圆时仍拒报孔径。这些是对模拟设计真值的误差，不代表实物精度；找正管轴也不能确定两个缺陷是否在同一材料位置。[看找正前后曲线、六组正反例和复现命令](reports/AXIS-GAUGE.md) · [下载 CPU 复核包](reports/axis-review.zip)。

## 定位局部减薄，而不只报告最薄点

把已对齐的基准管和待测管交给 `cone-fdk loss`，会得到每处减薄的范围、面积、剩余径向壁厚，以及峰值处内外壁各变化了多少。在同一根模拟管上，外壁削平和内孔局部扩大的设计深度是 0.850、0.750 mm，测得 **0.857、0.756 mm**。削平区虽然出现在展开图的两端，仍只算一处。对完好管独立加入两份计数噪声，同一 0.30 mm 门槛下未报出区域。[检测图、带噪对照、适用边界与运行方法](reports/WALL-LOSS.md) · [下载可用 CPU 逐点复核的样包](reports/wall-loss-demo.zip)。

## 换了装夹，能不能量出同一处缺陷加深了多少？

同一根**模拟管件**重新装夹后，按扫描仪坐标比较有 325 个单元超限；用四颗随件球标对齐后，14,760 个单元均未超限。把同一处外壁再切深 0.550 mm，测得壁厚减少 **0.54796 mm**。[复扫对照与复现命令](reports/REMOUNT-LOSS.md)

- **工装滑动：** 四颗球一起移动，拟合残差仍可能很小。`--witness` 用管件上的三个独立标记检查位姿；未通过就不生成壁图。[滑移对照](reports/BODY-WITNESS.md)
- **阈值变化：** 两次扫描各取三档边缘阈值，253 个单元在九种配对下始终超限，22 个随设置翻转。[逐格复核](reports/EDGE-REVIEW.md) · [连通区域与面积](reports/REGION-REVIEW.md) · [离线查看器](reports/region-console.html)
- **换试件：** 两种模拟管件的三处缺陷，标称设置下峰值与设计最大相差 0.0190 mm；临界窄桥则只在 3/9 种配对下连通。这些结果不能作为实物公差。[多试件对照](reports/MULTI-SPECIMEN.md) · [窄桥实验](reports/BRIDGE-STUDY.md)

批量复核可[中断续算](reports/STRIPED-BATCH.md)，有争议的位置可单独导出[跨接缝选区](reports/PERIODIC-ROI.md)。

## 扩展实验：在金属 CT 中逐孔测量

[孔隙测量分支](https://github.com/EtherealBian/cone-fdk/tree/measurement-lab/pores)的输入是已重建的钴铬合金 CT。给定掩膜，CUDA 程序能逐孔计算体积、等效直径和质心。从灰度中自动找准孔边界则仍有误差：候选 E 在 2、3、5 号开发试件分别通过 **10/11、6/11、7/11** 项，最差试件的匹配孔体积误差 P95 为 **28.16%**，门槛为 **15%**。4、6 号保留件未启用；该实验不包含在稳定发布包内。

[三件结果与原始档重放](https://github.com/EtherealBian/cone-fdk/blob/7b9d4bfc9c1ea8fe2750baa1a2b8154bfd09b418/pores/RETAINED-BASELINE.md) · [看同一切片上两个孔如何被连成一个](https://etherealbian.github.io/work/cone-fdk.html#pore-pair) · [四件外部试件的不同数字边界](https://github.com/EtherealBian/cone-fdk/blob/7b9d4bfc9c1ea8fe2750baa1a2b8154bfd09b418/pores/EXTERNAL-REFERENCE-AGREEMENT.md)

## 从源码运行

在 DGX Spark 上，按 [Spark 原生 CUDA 构建与运行步骤](platform/spark/QUICKSTART.md#run-native-cuda-from-a-fresh-source-checkout) 从源码运行。下面的 PowerShell 命令用于 Windows。

```powershell
git clone https://github.com/EtherealBian/cone-fdk.git
cd cone-fdk
# 按下节链接安装锁定依赖，然后在 x64 编译器环境中构建
.\Build.ps1
.\Demo.ps1                 # 离线：90 个实测角，原生 / ASTRA 对照，输出阅片器和 MetaImage
.\Run-Acceptance.ps1       # 重编译 + 回归 + 精度 + 性能 + 超显存 + 全部 720 个实测角
```

演示输出 `reports/walnut-viewer.html` 可直接在浏览器打开，无网络服务；`data/walnut-corrected-256.mhd` 和同目录 `.f32` 是保留定量信息的完整体。全量测试另外生成 `data/full-corrected-512.mhd`、对应 `.f32` 和 `reports/full/walnut-viewer.html`。HTML 内置 81 张固定显示窗的轴向预览，不是全部切面，也不应用来量灰度。

## 从发布包到新环境

Windows x64 用户可下载 v0.1.0 发布包，用 Python 3.12 安装锁定依赖，再运行 `Demo.ps1`。发布包包含 EXE、PTX 和 90 角演示数据，不含 NVIDIA DLL。完整验收还会下载约 4.24 GB 的原始扫描，建议预留 **25 GiB** 磁盘。[安装命令、驱动要求与发布包校验](README.zh-CN.md#从发布包到新环境)

## 自有扫描如何接入

`scan.geom` 是单行 ASCII，依次为标记、视角数、探测器行/列数、U/V 像素间距、源到等中心距离、源到探测器距离、起始角（弧度）、旋转符号（±1）。例：

```text
CFDK1 720 296 280 0.4 0.4 210.66 553.74 0 1
```

`scan.f32` 是小端 float32 的**线积分**（不是未校正强度），连续布局 `[角度,v,u]`，索引 `u` 最快。只接受完整均匀圆轨迹，第 `i` 角为 `start + sign * 2πi/views`；不要把 360° 终点重复输入。输出 `[z,y,x]`，每轴 `n`，体素中心位置 `(index-(n-1)/2)*fov/n`。距离与 FOV 用 mm，输出衰减用 mm⁻¹。包装器仅把 wheel 的 CUDA DLL 路径加入子进程环境；独立运行 EXE 时让 CUDA 12 的 cuFFT 及其依赖可被 Windows 加载器找到。

```powershell
# 在已运行 Demo 的目录；不加 --astra 默认执行自研 EXE
.\.venv\Scripts\python.exe scripts/reconstruct.py data/walnut-corrected data/my-volume.f32 --n 384 --fov 40 --budget 512 --batch 16
# 已配置 NVIDIA DLL 路径的独立终端，同一工作不需要 Python
.\build\cone-fdk.exe --ptx build/kernels.ptx --geom data/walnut-corrected.geom --input data/walnut-corrected.f32 --output data/my-volume.f32 --n 384 --fov 40 --budget 512 --batch 16
```

CLI 成功时 stdout 输出 JSON；失败返回非零并将错误写入 stderr。校验长度、几何、有限性和路径别名；同目录临时文件完整写入并 flush、显存边界校验后才替换目的文件。显存预算单位 MiB，不等于整个进程或驱动的峰值。`--host-memory auto`（默认）分别尝试锁页输入和输出暂存区，每块上限 8 MiB；超过上限或锁页内存不足时使用普通主机内存。`--host-memory pageable` 禁用锁页。`host_pinned_bytes` 单列实际锁页主机字节，不计入设备缓冲峰值。

## 哪些部分值得读

| 路径 | 解决的问题 |
|---|---|
| `src/kernels.cu` | 先乘 `DSD/sqrt(DSD²+u²+v²)`；U 方向零填充到至少两倍探测器宽度，按离散 Ram-Lak 滤波。以体素为中心双线性采样；探测器外零延拓。每线程算 16 个相邻 Z，保留原四层组的坐标舍入顺序。用已释放使用权的 FFT 缓冲暂存水平插值系数；两侧零哨兵维持零延拓。 |
| `src/main.cpp`、`src/gpu.hpp` | 按预算规划角度批与 Z 分块，显式管理 cuFFT 工作区。分块复用缓冲，保持视角累加次序，每个 slab 付出重读重滤的代价；含 CUDA Event 计时、边界哨兵与占用计数。 |
| `src/files.hpp` | Windows Unicode 文件、身份别名检测、大体积分段写出及事务式发布。 |
| `tests/test_spatial_oracle.py` | 小体积独立 FP64 空间域参考：直接卷积而非 FFT，用射线与平面的交点逐体素核验 GPU 实现。 |
| `scripts/phantom.py`、`scripts/accept.py` | 用解析射线与有限圆柱/球相交生成独立真值；核验孔位、孔径、衰减、离面夹杂及全体积误差。 |
| `scripts/full_walnut.py`、`scripts/prepare_walnut.py` | 按原始元数据构造真实几何。原始投影已校正暗场/平场；每帧原图左上 128×128 算空气基准，水平循环左移 5 原始像素，8×8 强度均值后取负对数。另保留未移位的对照。 |
| `scripts/benchmark.py`、`scripts/stress.py` | 新进程交替测本机 GPU 与 ASTRA；压力测试真正落盘超显存体并跨接缝用 ASTRA 独立核查五层。 |

## 适用边界

这是面向工业 CT 算法与 GPU 工程的可复现原型，**不是经过计量检定的工业检测设备或医疗软件**。仅支持居中、平板、等角间隔的单圈锥束。默认整件输出为各向同性立方体；定向切片和层栈可分别指定各轴间距。圆轨迹 FDK 在离中平面处是近似；没有螺旋、短扫描/Parker、散射与硬化校正、金属伪影消除或实际缺陷检出认证。512³ 网格间距不等于系统真实空间分辨率；与 ASTRA 一致也不代表实物尺寸真值。许可证、数据署名及变换说明见 [NOTICE](NOTICE.md)。

---

## English

[简体中文](#zh-cn) · **English**

**ConeFDK · Single-GPU cone-beam CT**

ConeFDK turns 720 measured X-ray projections of a walnut into a **512³ volume** on an NVIDIA DGX Spark (GB10). The C++/CUDA engine weights each projection, filters detector rows with cuFFT, and backprojects into slabs that fit the GPU memory budget. Python prepares the input and checks the output; the native executable runs without it.

[Project walkthrough](https://etherealbian.github.io/work/cone-fdk.html) · [Windows release v0.1.0](https://github.com/EtherealBian/cone-fdk/releases/tag/v0.1.0) · [Download the offline slice viewer](https://github.com/EtherealBian/cone-fdk/releases/download/v0.1.0/walnut-512-viewer.html)

![Axial sweep through a walnut reconstructed from 720 measured views](reports/walnut-sweep.gif)

*The image is a sweep through measured data, not a rendering of a synthetic walnut. The downloadable HTML lets you scrub 81 preview slices without installing CUDA. Data: [Alexander Meaney / University of Helsinki](https://doi.org/10.5281/zenodo.6986012), CC BY 4.0; processing and attribution in [NOTICE](NOTICE.md).*

### Measured on DGX Spark: one walnut, three reconstruction grids

| Check | Result | Evidence |
|---|---|---|
| 720 views, 512³ reconstruction | Two native runs produced identical bytes. Against a separate ASTRA FDK run on Spark, full-volume relative L2 was **0.02899%**, without registration or intensity fitting. | [Native repeats](reports/gb10/walnut-native.json) · [ASTRA comparison](reports/gb10/walnut-astra.json) |
| One projection pass for three grids | A full overview and two finer regions took **1.743 → 0.643 s** median native process time over six pairs. Each output matched its standalone reconstruction byte-for-byte. | [Paired trials and timing boundary](reports/gb10/multigrid-timing.json) |
| Resume after a process kill | On the 720-view walnut, two committed slabs were reused and three were computed after restart. All **512 MiB** matched the uninterrupted reconstruction byte-for-byte. | [Recovery receipt](reports/gb10/slab-recovery.json) |
| Review without a GPU | Two additional measured regions ship with both native and ASTRA voxels. Recompute the per-plane differences across **138,948 voxels** using standard Python. | [Quickstart](platform/spark/QUICKSTART.md) · [Sample data](examples/gb10-walnut) |
| Where do the two solvers differ most? | Across 512 axial planes, the first maximum is at **(367, 318, 256)**. The offline snapshot includes both raw peak slices: remeasure all 262,144 pairs without a GPU. | [Peak profile and reproduction](platform/spark/QUICKSTART.md#offline-physical-peak) · [Queue export checks](reports/gb10/physical-peak-snapshot.json) |

**Two physical specimens, three scan conditions.** The completed batch adds a pine cone reconstructed on a 256³ grid: native versus ASTRA relative L2 **0.07314% at 90 views** and **0.02658% at 720 views**. Two direct native crops per condition matched their dense counterparts byte-for-byte. The view counts describe one pine cone; neither solver supplies physical dimensional truth. [Final batch](reports/gb10/physical-batch-complete.json) · [Recheck all three peak planes without a GPU](platform/spark/QUICKSTART.md#completed-physical-batch). Unless explicitly labeled Spark, benchmarks below retain their historical platforms and timings.

**Spark migration:** 929 Spark tests passed. Of the same 707 historical cases, 702 pass; five require original-path write or rename denial. [Build the measurement tools and review exactly what passed](platform/spark/QUICKSTART.md#spark-migration-acceptance).

### How sensitive is a wall-loss footprint to its edge threshold?

In a **simulated** changed specimen, all nine reference/inspection threshold pairs found one connected region, but its footprint ranged from **253 to 275 cells**. The repeat control found none. A separate invalid-edge control retains its original four pairs and marks unknown cells explicitly. Two native CUDA backends on Spark reproduced the archived wall and classification files byte-for-byte. [Recompute the 22 cylindrical label grids without a GPU](platform/spark/QUICKSTART.md#offline-threshold-review) · [Pair-by-pair results](reports/gb10/directory-edge-qualification.json).

### Reconstruct just the region you need

`--roi` reconstructs a box directly from the projections, preserving the parent scan’s coordinates and voxel pitch. A 128 × 128 × 96 walnut region used **6 MiB** instead of 512 MiB; all six runs matched the corresponding full-volume crop byte-for-byte. [ROI example and commands](reports/ROI.md)

Use [`--roi-list`](reports/MULTI-ROI.md) to share filtered projections across regions. In the archived Windows benchmark, [`--planner min-passes`](reports/ROI-PLANNER.md) cut projection reads from three passes to one for six regions at a 48 MiB budget. Median process time fell from 0.590 to 0.304 s. These options require a source build.

### Inspect an oblique cross-section

`--plane` reconstructs a slice at the position, orientation and pixel spacing you specify. A tilted 320 × 288 walnut slice occupies **360 KiB**, without writing the full volume. [Slice examples and FP64 comparison](reports/DIRECT-PLANES.md)

Use `--stack` to export parallel slices as one NRRD with scanner-space coordinates. All 21 tilted walnut slices matched the individual plane jobs byte-for-byte, with one projection pass instead of 21. [Animated stack and commands](reports/OBLIQUE-STACK.md)

### Unwrap a cylinder through the scan

A flat slice cuts a ring twice. `--cylinder` follows the ring for a full turn instead. On the measured walnut, 11 radii from 7 to 17 mm share one read of the 720 projections. In the [offline atlas](reports/walnut-cylinder-atlas.html), change radius, move the seam, or click two points to measure the shortest path **on that cylinder**, including across the image edges. [Screenshot, coordinate convention, accuracy checks and CLI](reports/CYLINDER-ATLAS.md).

![Measured walnut, concentric sheets unwrapped from 7 to 17 mm](reports/cylinder-sweep.gif)

### Find the thin side of a tube

`cone-fdk wall` measures the inner radius, outer radius and **radial wall thickness** at each azimuth and height of a cylindrical reconstruction. On two simulated eccentric bores, the thinnest 15° sectors landed at the designed **358° and 134°**. Maximum pointwise thickness errors were **0.01322 mm and 0.01065 mm**. Adding a separated liner made every ray cross two walls; the engine rejected all 6,120 instead of reporting a spurious thickness. [Results, raw profile evidence and reproduction](reports/RADIAL-WALL.md). These are simulation results, not accuracy claims for a physical pipe. Available from a source build.

### Measure bore diameter and eccentricity after repositioning

`cone-fdk bore` fits an inner and outer circle independently at each height of a wall map. It reports both diameters, their centers in scanner coordinates, and the minimum gap between the fitted circles. In a simulated remount, the maximum change in outer diameter was **0.01696 mm** at the original sampling and **0.00788 mm** with both detector and radial pitches halved (gate: 0.012 mm). The latter needs new, finer projections; it is not a software-only improvement to an existing scan. [Compare all four settings, inspect a rejected ellipse, and run the gauge](reports/BORE-GAUGE.md).

### Gauge a tube mounted at an angle

A circular tube cut in the wrong plane looks elliptical. The source-build `axis` workflow estimates the outer centerline, then reconstructs **again from the original projections**, perpendicular to that line. On the 8° and 12° simulated mounts, the unchanged circle gauge went from **0/25 to 25/25 accepted sections per scan**. Across four circular cases including one with counting noise, maximum diameter error was **0.0010905 mm** against design. An oval outer wall stopped the workflow; an oval bore on a round outer wall still failed the bore gauge. This locates a line, not a material landmark for aligning loss maps. [Before/after profiles, limits and commands](reports/AXIS-GAUGE.md) · [CPU-checkable example](reports/axis-review.zip).

### Locate local wall loss, including across the unwrap seam

Given a registered baseline and inspected tube, `cone-fdk loss` reports **where material was lost**: the footprint, maximum loss, remaining radial wall, and the separate inner/outer changes at its peak. In a simulated tube with an outside flat and an inside relief, their peak losses measured **0.857 and 0.756 mm** against designs of 0.850 and 0.750 mm. The outside flat appears at both ends of the unwrapped image but counts only once. A separately noise-sampled pristine pair produced zero regions at the same 0.30 mm threshold. [See the footprints, noise cases, limits and commands](reports/WALL-LOSS.md) · [Download a compact sample for CPU-only review](reports/wall-loss-demo.zip).

### Reinspect the same defect after a remount

Comparing an unchanged **simulated tube** in scanner coordinates flagged 325 cells after remounting. Aligning the scans with four fixture beads reduced that to zero out of 14,760. When the same flat was cut another 0.550 mm deeper, measured wall thickness fell by **0.54796 mm**. [Remount comparison and commands](reports/REMOUNT-LOSS.md)

- **Fixture slip:** Four beads can move together and still give a small fit residual. `--witness` checks three independent landmarks on the tube and withholds the wall maps if the pose fails. [Slip controls](reports/BODY-WITNESS.md)
- **Edge thresholds:** Across nine threshold pairs, 253 cells always exceeded the loss threshold and 22 changed classification. [Cell-level replay](reports/EDGE-REVIEW.md) · [Connected regions and areas](reports/REGION-REVIEW.md) · [Offline viewer](reports/region-console.html)
- **Other geometries:** Across three defects on two simulated tubes, the largest peak deviation from design at nominal settings was 0.0190 mm. A narrow bridge stayed connected in only three of nine threshold pairs. These are simulation results, not physical tolerances. [Multi-specimen comparison](reports/MULTI-SPECIMEN.md) · [Narrow-bridge study](reports/BRIDGE-STUDY.md)

For larger reviews, [resume an interrupted batch](reports/STRIPED-BATCH.md) or [export the cells around a disputed location](reports/PERIODIC-ROI.md), including across the cylindrical seam.

### A second use: measuring pores in metal

The [measurement lab](https://github.com/EtherealBian/cone-fdk/tree/measurement-lab/pores) starts from **already reconstructed** cobalt-chromium scans. Its CUDA counter measures connected pores in a supplied binary mask. Finding the pore boundary from grayscale is a separate experiment: on development specimens 2, 3 and 5, the retained model E passed **10/11, 6/11 and 7/11** criteria. Its worst matched-pore volume error (P95) was **28.16%**, above the **15%** limit. The two reserved specimens remain unopened. This work is not part of the stable reconstruction release.

[Compare the three specimens and replay E](https://github.com/EtherealBian/cone-fdk/blob/7b9d4bfc9c1ea8fe2750baa1a2b8154bfd09b418/pores/RETAINED-BASELINE.md) · [Inspect a pore in the source slices](https://etherealbian.github.io/work/cone-fdk.html#pore-pair) · [Compare digital boundaries from four external specimens](https://github.com/EtherealBian/cone-fdk/blob/7b9d4bfc9c1ea8fe2750baa1a2b8154bfd09b418/pores/EXTERNAL-REFERENCE-AGREEMENT.md)

### Try it

- **Build and run on DGX Spark:** follow the [native CUDA quickstart](platform/spark/QUICKSTART.md#run-native-cuda-from-a-fresh-source-checkout).
- **Just inspect the result:** download the HTML viewer above and open it in a browser. No installation or network connection required to view the slices.
- **Run the offline 90-view demo:** extract `ConeFDK-win-x64.zip` from the release. In that directory run the commands below on a compatible Windows x64 NVIDIA machine. The release includes the executable, PTX and measured fixture, not NVIDIA DLLs. Installing dependencies requires internet access.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win.lock
.\Demo.ps1
```

- **Build from a source clone:** install the same locked dependencies, open an x64 MSVC Developer PowerShell (or set `CXX` to a 64-bit MinGW compiler), then run `.\Build.ps1` before `.\Demo.ps1`. `.\Run-Acceptance.ps1` also exercises the 720-view dataset and the disk-backed stress case. Allow about 25 GiB free. See the [input contract and CLI examples (in Chinese)](#自有扫描如何接入).
- **Reproduce the optimization comparison:** the release includes a compact `ConeFDK-ab-baseline.zip` with the previous native executable, PTX, source and reference output hashes. See [benchmark protocol](reports/OPTIMIZATION.md). Its wrapper is a new ZIP, not the archived full development release; the native binary hashes match the original benchmark.

### Where the engineering is

| Read | Why |
|---|---|
| [CUDA kernels](src/kernels.cu) | Voxel-driven backprojection, Z-register tiling and interpolation coefficients packed into the former FFT scratch region. |
| [Memory planner and runtime](src/main.cpp) | Selects angle batches and Z slabs under a byte budget. Smaller budgets trade rereading/refiltering for lower residency. |
| [Windows file I/O](src/files.hpp) | Unicode paths, input/output alias checks, and transactional publishing after successful validation and flush. |
| [FP64 spatial oracle](tests/test_spatial_oracle.py) · [Acceptance gates](reports/ACCEPTANCE.md) | Direct convolution and independently constructed projection geometry, alongside analytic phantoms and real data. |

**Scope:** centered, circular, full-scan cone-beam geometry with a flat detector. Experimental reconstruction software, not a calibrated industrial inspection system or clinical device. Agreement with ASTRA is not dimensional ground truth. The reported memory budget excludes driver/context allocations. No claim that Compute Sanitizer passed; its attachment limitation is documented in the [test report](reports/RESULTS.md).

Original code: [MIT](LICENSE). Data and external tools retain their own licenses; see [NOTICE](NOTICE.md).
