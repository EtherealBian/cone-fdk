# ConeFDK · 单 GPU 锥束 CT 重建

在 NVIDIA DGX Spark（GB10）上，用 720 幅核桃 X 射线实测投影重建 **512³ 体数据**。原生 C++/CUDA 引擎完成锥束加权、cuFFT 分批滤波和按显存预算分块的反投影。Python 用来准备输入和独立校验；重建程序本身不需要 Python。

[English](README.md#english) · [个人主页项目详解](https://etherealbian.github.io/work/cone-fdk.html) · [Windows 稳定版 v0.1.0](https://github.com/EtherealBian/cone-fdk/releases/tag/v0.1.0) · [下载离线切片浏览器](https://github.com/EtherealBian/cone-fdk/releases/download/v0.1.0/walnut-512-viewer.html)

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
# 按下节安装锁定依赖；在 x64 编译器环境中先从源码构建
.\Build.ps1
.\Demo.ps1                 # 离线：90 个实测角，原生 / ASTRA 对照，输出阅片器和 MetaImage
.\Run-Acceptance.ps1       # 重编译 + 回归 + 精度 + 性能 + 超显存 + 全部 720 个实测角
```

演示输出 `reports/walnut-viewer.html` 可直接在浏览器打开，无网络服务；`data/walnut-corrected-256.mhd` 和同目录 `.f32` 是保留定量信息的完整体。全量测试另外生成 `data/full-corrected-512.mhd`、对应 `.f32` 和 `reports/full/walnut-viewer.html`。HTML 内置 81 张固定显示窗的轴向预览，不是全部切面，也不应用来量灰度。

## 从发布包到新环境

Windows x64、支持 compute_80 PTX 和 CUDA 12.9 PTX JIT 的 NVIDIA GPU/驱动。发布包附已编译的 EXE/PTX，**不附第三方 DLL、Python、自称可用的模拟 GPU 或扫描仪**。使用 Python 3.12 x64，并允许从 Python 包源安装锁定的依赖。若需重编译，准备 x64 MSVC Developer PowerShell 或 MinGW-w64；本次测试使用 MinGW。本机还验证了两次从源重编所得 EXE/PTX 的 SHA-256 一致，见 [构建记录](reports/build-repro.json)；不同编译器或驱动未作这种承诺。验证已解压文件：`Get-FileHash` 可对照随包的 `SHA256SUMS`，或运行 `py -3.12 scripts/release.py verify .`。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win.lock
.\Demo.ps1
.\Run-Acceptance.ps1 -SkipBuild            # 验证交付的二进制
.\Run-Acceptance.ps1 -CoreOnly -SkipBuild  # 较短；跳过超显存与全角测试
```

默认完整验收需要按 Zenodo 记录下载约 4.24 GB 原始归档，已有归档会核验 MD5 并复用；建议预留 **25 GiB** 可用磁盘。`1296³` 压力测试的临时原始体核验后删除。该固定尺寸用例特意断言输出大于本机物理显存，换到显存更大的 GPU 不能原样证明“超显存”，应调整压力体和取样几何。WDDM 遥测基于本地 Windows 显示驱动栈，其他系统、卡型与工具链不在这次验收范围。

如需自行打包：完成全量验收，并按 [A/B 复现步骤](reports/OPTIMIZATION.md) 生成本次二进制的性能与跨版本输出记录；用 `.\Build.ps1 -VerifyRepeat` 自动双编并记录二进制指纹（只有指纹一致才通过）。运行 `python scripts/release.py build`，再用同一验收环境运行 `python scripts/check_release.py dist/ConeFDK-win-x64.zip`。后者在临时中文路径内校验全部成员，删掉已有演示图，重新从随包投影生成体数据并逐张校验 81 张 PNG。结果写到包旁的 `verification.json`。

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
