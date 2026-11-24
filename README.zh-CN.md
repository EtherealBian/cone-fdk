# ConeFDK · Windows 单 GPU 锥束 CT 重建

在一台 Windows 笔记本上，用 720 幅核桃 X 射线实测投影重建 **512³ 体数据**。原生 C++/CUDA 引擎完成锥束加权、cuFFT 分批滤波和按显存预算分块的反投影。Python 用来准备输入和独立校验；重建程序本身不需要 Python。

[English](README.md) · [个人主页项目详解](https://etherealbian.github.io/work/cone-fdk.html) · [Windows 稳定版 v0.1.0](https://github.com/EtherealBian/cone-fdk/releases/tag/v0.1.0) · [下载离线切片浏览器](https://github.com/EtherealBian/cone-fdk/releases/download/v0.1.0/walnut-512-viewer.html)

![720 角真实投影重建的核桃轴向切层](reports/walnut-sweep.gif)

动图中的壳、隔膜和核仁均来自实测数据：[Alexander Meaney / 赫尔辛基大学](https://doi.org/10.5281/zenodo.6986012)，CC BY 4.0。离线浏览器可拖动查看 81 张预览切片，无需 CUDA。[数据署名与处理说明](NOTICE.md)

## 重建实测

| 问题 | RTX 5060 Laptop 上的结果 | 可复核记录 |
|---|---|---|
| 与独立实现相差多少？ | 720 角、512³ 输出，与 ASTRA FDK 的全体积相对 L2 差异 **0.02899%**，未拟合强度或配准。 | [数值对照](reports/RESULTS.md) |
| 优化实际省了多少时间？ | 与上一版原生引擎同机对比，整进程中位数 **2.392 → 2.009 秒**，六组配对输出逐字节相同。 | [逐次计时](reports/native-ab.json) |
| 超过显存还能重建吗？ | 以 **512 MiB 引擎设备缓冲预算**写出 1296³ FP32 体数据，落盘 8.71 GB；此规模用 ASTRA 抽查五层。 | [压力测试](reports/oversize.json) |

## 扩展实验：在金属 CT 中逐孔测量

[孔隙测量分支](pores/README.md)的输入是已重建的钴铬合金 CT。给定掩膜，CUDA 程序能逐孔计算体积、等效直径和质心。从灰度中自动找准孔边界则仍有误差：候选 E 在 2、3、5 号开发试件分别通过 **10/11、6/11、7/11** 项，最差试件的匹配孔体积误差 P95 为 **28.16%**，门槛为 **15%**。4、6 号保留件未启用；该实验不包含在稳定发布包内。

[三件结果与原始档重放](pores/RETAINED-BASELINE.md) · [看同一切片上两个孔如何被连成一个](https://etherealbian.github.io/work/cone-fdk.html#pore-pair) · [四件外部试件的不同数字边界](pores/EXTERNAL-REFERENCE-AGREEMENT.md)

## 从源码运行

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

这是面向工业 CT 算法与 GPU 工程的可复现原型，**不是经过计量检定的工业检测设备或医疗软件**。仅支持居中、平板、等角间隔的单圈锥束、正方体各向同性输出。圆轨迹 FDK 在离中平面处是近似；没有螺旋、短扫描/Parker、散射与硬化校正、金属伪影消除或实际缺陷检出认证。512³ 网格间距不等于系统真实空间分辨率；与 ASTRA 一致也不代表实物尺寸真值。许可证、数据署名及变换说明见 [NOTICE](NOTICE.md)。
