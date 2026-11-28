# ConeFDK · 胸部锥束 CT 三维重建

在 NVIDIA DGX Spark 上，用自写 C++/CUDA [FDK](https://doi.org/10.1364/JOSAA.1.000612) 将 **一例 POPI 胸部实测投影**重建为静态三维体积。程序逐幅读取 Elekta 扫描角度及探测器偏移，按相邻角距加权；每 16 幅复用一个 cuFFT plan 做离散 Ram-Lak 滤波，再用双线性插值反投影。正式重建只依赖 CUDA Runtime 和 cuFFT。

| 投影 × 探测器 | 输出体积 | 平台 | 反投影 GPU 中位数 | CLI 端到端中位数 | 全体积相对 L2 |
|---|---|---|---:|---:|---:|
| 644 × 512 × 512 | 256³，1.25 mm 等距 | NVIDIA GB10 · CUDA 13.0.88 | 46.23 ms | 998.56 ms | 0.01761% |

Release；预热一次后，分别取五次的中位数。GPU 时间是 41 批反投影 kernel 的 CUDA event 时间之和；CLI 墙钟从启动到输出关闭并退出，包含初始化、读写和传输，不含下载、格式转换和参考重建。重复读写使用系统页缓存，不代表冷盘或强制落盘性能。

## 在 Spark 上复现

在此源码目录执行。需要 CMake ≥ 3.24、C++17、CUDA Toolkit、Python ≥ 3.11。`CASE` 为仓库外的私有目录；下载只涉及这一例的投影、逐帧记录和扫描几何。已下载的原文件会校验后复用。

```bash
set -eu
umask 077
export PATH=/usr/local/cuda/bin:$PATH
export CASE="${CASE:-$HOME/cone-fdk-popi-case}"
mkdir -p "$CASE"
api=https://data.kitware.com/api/v1/item
for pair in projections.tgz:5be99af88d777f2179a2e144 FRAME.DBF:5be99a068d777f2179a2cf4f geometry.xml:5be99a078d777f2179a2cf5b; do
  file="$CASE/${pair%:*}"
  if [ ! -f "$file" ]; then
    curl -fsSL "$api/${pair#*:}/download" -o "$file.part"
    mv "$file.part" "$file"
  fi
done
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --parallel
python3 -m venv build/venv
build/venv/bin/python -m pip install --no-cache-dir numpy==2.5.5 itk-rtk==2.7.0.post1
build/venv/bin/python tools/prepare.py "$CASE"
build/venv/bin/python tools/verify.py ./build/cone-fdk "$CASE"
```

最后一条命令打印五次原始计时，使用独立的 **RTK CPU FDK** 读取发布方原始 `geometry.xml` 并比较全体积。若只需重建，运行 `./build/cone-fdk "$CASE/views.geom" "$CASE/views.f32" "$CASE/volume.f32"`。结果是小端 FP32 `[z,y,x]`；RTK 固定坐标系中 Y 为转轴，原点位于等中心，首体素中心为 (−159.375, −159.375, −159.375) mm。

## 数据与验收口径

- **来源与使用：** [RTK 官方 POPI 投影示例](https://docs.openrtk.org/en/latest/applications/rtkfourdconjugategradient/README.html)，[POPI 研究使用与引用说明](https://github.com/open-vv/popi-model/blob/master/popi-model.md)。数据来自法国里昂 Léon Bérard Cancer Center 与 CREATIS Laboratory。引用 J. Vandemeulebroucke、D. Sarrut、P. Clarysse，《The POPI-model, a point-validated pixel-based breathing thorax model》，ICCR 2007。三个公开对象以 SHA-256 固定版本；本仓库不分发患者投影、数据库或重建体积。
- **输入域与几何：** HIS 是 16 位透射强度的反码；依据 [RTK 的 Elekta 转换](https://github.com/RTKConsortium/RTK/blob/v2.7.0/include/rtkElektaSynergyRawLookupTableImageFilter.hxx) 处理保留码后，以空气参考 65,536 取一次负对数。保留全部探测器像素。文件编号逐幅对应帧表，XML 中的角度及水平、垂直偏移与帧表对应。探测器间距 0.8 mm，源至等中心 1,000 mm、源至探测器 1,536 mm。所有视角合入一个静态体积。
- **正确性：** 参考和本实现使用相同线积分、角距权重、锥束权重、未加窗 Ram-Lak 滤波、体素网格和物理尺度。验收前将 `‖CUDA−RTK‖₂ / ‖RTK‖₂` 上限定为 **0.1%**，给 FP32 FFT、插值和 644 幅累加留出数值差异余量。比较整个 320 mm 立方视野，不裁剪、不配准、不拟合强度；另在三个固定轴的非中心切片排除反向、镜像与转置。
- **边界：** 结果限于这一例和采集视野。呼吸运动与探测器截断仍可造成伪影，输出未经 HU 标定。数值一致性不代表临床诊断准确性。
