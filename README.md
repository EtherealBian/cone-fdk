# cone-fdk — Single-GPU Cone-Beam FDK Reconstruction in CUDA

在 **NVIDIA DGX Spark（GB10）** 上，从核桃的 720 幅投影重建 **512³ FP32** 体数据。当前单网格 CLI 的端到端中位数为 **1.504 s**；同一工作负载中反投影的有效带宽为 **127.4 GB/s**，实际 DRAM 峰值占比 `<待实测>`。

[滤波 kernel](src/filter.cu) · [反投影 kernel](src/backproject.cu) · [性能记录](profiles/) · [优化决策](docs/optimization-log.md)

## Quick Start

需要 CMake ≥ 3.24、支持本机 GPU 的 CUDA Toolkit（含 cuFFT）和 C++17 编译器。在仓库根目录执行；无须下载数据。已实测 GB10，未做跨 GPU CI。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel
./build/cone-fdk
```

无参数运行内置小体模；[真实投影示例](experimental/README.md#运行随仓核桃样例)使用随仓的 90 幅投影。

## Performance at a Glance

| Stage | Time (ms) | % of total | Achieved BW¹ | % of DRAM peak |
|---|---:|---:|---:|---:|
| Filter (cuFFT) | 61.02 | 3.41% | 11.76 GB/s | `<待实测>` |
| Backprojection | 379.19 | 21.17% | 127.42 GB/s | `<待实测>` |
| Total（含 I/O） | 1,791.57 | 100% | — | — |

同一次 Nsight Systems 跟踪：720 角、296 × 280 探测器、16 角/批。前两行是 GPU kernel 活跃时间，分母是被跟踪进程的墙钟时间；未列入的时间包括初始化、数据传输和文件 I/O。首页的 1.504 s 是**不挂 profiler** 的五次进程中位数。¹ 有效带宽只计阶段边界的逻辑字节，不是显存总线流量。[逐项计时、测量命令及带宽口径](profiles/README.md)。

## Kernel Design

- **Filter (cuFFT)**：一份 C2C plan 批处理探测器行，并复用于所有角度批次的正、逆变换。逆变换后的工作区改存 `(左值, 右值−左值)`，供反投影插值复用。[代码](src/filter.cu) · [NCU 读数](profiles/filter_ncu.txt)
- **Backprojection**：每线程负责一个 `(x,y)` 及最多 16 个 z 体素；它们共享投影列与几何权重。角度与插值表存放在只读访问的全局内存。首个满批次 `back` 测得 **72 registers/thread，58.02% achieved occupancy**。[代码](src/backproject.cu) · [NCU 读数](profiles/backproject_ncu.txt)

## Optimization Log

历史三网格实现共用一次滤波，批次数 **135 → 45**；六组配对中，进程中位数由 **1,743 → 643 ms（2.71×）**。这是三网格调度的收益，不是当前单网格的加速比。[四项优化及取舍](docs/optimization-log.md)

## Correctness

两项 CTest 覆盖示例和独立 FP64 空域参考。新旧程序在核桃 720 角及松果 90/720 角上，三个完整体积均逐字节相同；同一核桃的历史独立 ASTRA 全体积相对 L2 差异为 **0.02899%**。[输入、输出指纹与复核方式](experimental/README.md)。

![从 720 幅真实投影重建的核桃轴向切片序列](examples/walnut-sweep.gif)

*投影：Alexander Meaney / University of Helsinki，CC BY 4.0；动图为重建结果。[署名](NOTICE.md)*

## Limitations & Future Work

主 API 仅做圆轨迹、等角采样的立方稠密网格，完整体积须能驻留 GPU。反投影是最长的 GPU 阶段；下一步值得对比不同探测器尺寸下的 z 分块，而不是仅凭 occupancy 修改默认值。

## Repository Layout

`src/` 核心 · `bench/` 阶段微基准 · `profiles/` 实测 · `docs/` 设计决策 · `examples/` CLI 和成像结果 · `tests/` 正确性；`experimental/` 仅留真实样例、输出指纹及两份对照证据。
