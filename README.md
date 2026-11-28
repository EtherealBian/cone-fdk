# ConeFDK · 单 GPU 锥束 CT 重建

用 CUDA C++ 和 cuFFT 将 **720 幅核桃 X 射线投影**重建为 **512³ FP32 体积**。在 NVIDIA DGX Spark（GB10）上，从读取投影到写完体积，中位耗时 **1.545 s**；与同机 ASTRA 独立重建相比，全体积相对 L2 误差为 **0.02899%**。

![从 720 幅实测投影重建的核桃轴向切片](examples/walnut-sweep.gif)

## 运行随仓样例：90 幅投影

需要 CMake ≥ 3.24、C++17、CUDA Toolkit（含 cuFFT）及装有 NumPy 的 Python 3。以下命令不需要联网；输出为 `[z,y,x]` 顺序的小端 FP32，x 变化最快。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel
python3 -c "import numpy as np; np.load('examples/walnut/corrected.npz')['views'].tofile('build/walnut90.f32')"
./build/cone-fdk examples/walnut/corrected.geom build/walnut90.f32 128 40 build/walnut128.f32
ctest --test-dir build --output-on-failure
```

## CUDA 怎样处理一批投影？

- [滤波](src/filter.cu)：每批 16 个角度，沿探测器行做锥束预加权和离散 Ram-Lak 滤波。所有批次的正、逆变换共用一个 cuFFT plan。逆变换结束后，同一块 FFT 工作区改存相邻采样的左值和差值，供反投影插值使用。
- [反投影](src/backproject.cu)：一个线程负责一个 `(x,y)` 和最多 16 层 z；同列体素共用投影横坐标与几何权重，在寄存器中累加。只支持圆轨迹、等角采样和能完整驻留 GPU 的单个立方网格。调用方提供已校正、有限的 FP32 投影。

## GB10 实测：720 幅投影 → 512³

| 测量对象 | 时间 | 口径 |
|---|---:|---|
| 完整 CLI | **1,544.55 ms** | 不挂 profiler；预热一次，五个独立进程取中位数，含文件读写 |
| 滤波 | 60.10 ms | 单次 Nsight Systems 跟踪；合计 45 批的滤波 kernel 和 cuFFT 时间 |
| 反投影 | 401.30 ms | 同一条跟踪；合计 45 次反投影 kernel |

反投影按每批读、写一次体数据计算的有效带宽为 **120.41 GB/s**，不是实测 DRAM 带宽。[五次原始计时、逐 kernel 用时、NCU 占用率及复测命令](profiles/dense-timing.txt)。

**正确性。** [独立 FP64 测试](tests/dense.py)用空域卷积计算参考值，覆盖两个旋向、探测器边缘、末尾不足一批的角度和不足 16 层的体积尾块。对真实核桃，使用相同投影与几何在 GB10 上另跑 ASTRA `FDK_CUDA`，直接比较全部 134,217,728 个体素，不做配准或灰度拟合。[ASTRA 对照结果](profiles/walnut-astra.json)。

**复现 720 角输入。** 从[原始核桃数据集](https://doi.org/10.5281/zenodo.6986012)下载 `20201111_walnut_projections_1.zip` 至 `_8.zip`，原样放入同一目录。安装 Pillow 后运行：

```sh
python3 examples/walnut/prepare720.py /path/to/zips build
./build/cone-fdk build/walnut720.geom build/walnut720.f32 512 40 build/volume.f32
```

预处理选取 0° 至 359.5° 的 720 幅不同投影；从原图取空气参考，水平校正 5 像素并做 8×8 分箱，最后归一化、取负对数。随仓的 90 幅样例从首幅开始，每隔 8 个角度取一幅，不用于上表计时。[数据与图像署名](NOTICE.md)。
