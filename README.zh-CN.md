# ConeFDK · Windows 单 GPU 锥束 CT 重建

**孔隙测量实验：** [查看钴铬试件的逐孔结果、误差和验收标准](https://github.com/EtherealBian/cone-fdk/tree/measurement-lab/pores)。下文是已发布的 CT 重建器；自动找孔精度尚未通过最终验收。

给定一圈等角间隔的平板 X 射线投影，重建可按毫米坐标读取的三维衰减体。**重建引擎是原生 C++ / CUDA**：手写锥束加权、有限离散 Ram-Lak 响应、反投影；只把一维批量 FFT 交给 NVIDIA cuFFT。Python 不在重建执行路径中，用来准备数据、调用独立的 ASTRA GPU 基线和做验收。

- [与上一发布版的同机 A/B](reports/OPTIMIZATION.md) · [逐次原始计时](reports/native-ab.json)
- [本机实测报告](reports/RESULTS.md) · [预先写定的验收标准](reports/ACCEPTANCE.md) · [720 角真实核桃浏览器](https://github.com/EtherealBian/cone-fdk/releases/download/v0.1.0/walnut-512-viewer.html)
- [真实数据校正前后及 ASTRA 对比](reports/walnut-full.png) · [三正交截面](reports/walnut-full-orthogonal.png) · [合成工件与离面夹杂](reports/phantom-512.png)

![720 角真实投影重建的核桃轴向切层](reports/walnut-sweep.gif)

[English overview](README.md) · [Windows 发布包](https://github.com/EtherealBian/cone-fdk/releases/tag/v0.1.0)

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



