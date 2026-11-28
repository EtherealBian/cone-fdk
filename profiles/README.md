# 复测 GPU 阶段

[各次计时和 kernel 汇总](dense-timing.txt) · [滤波 NCU 摘录](filter_ncu.txt) · [反投影 NCU 摘录](backproject_ncu.txt)。DGX Spark / NVIDIA GB10、CUDA 13.0.88、cuFFT 12.0、Nsight Compute 2025.3.1.0、Nsight Systems 2025.3.2，CMake Release。不要在 profiler 中测端到端中位数。

## 两个微基准

```sh
./build/bench_filter 256 16 9
./build/bench_backproject 256 16 9
```

参数依次为立方网格边长（也是此合成基准的探测器边长）、批内角数、计时次数。一次预热不计，取 CUDA Event 中位数；输入传输、清零及 plan 创建在计时区外。滤波统计 FP32 输入和 `float2` 表输出；反投影每批计 `8 × n³` 字节（读写体积各一次）。两者都不含内部临时流量、缓存重用或探测器 gather 的实际显存交易，**不能除以标称显存带宽当作峰值利用率**。

## Nsight Compute

下例前两行用于探索，不是计时。机器须允许访问性能计数器；没有权限会收到 `ERR_NVGPUCTRPERM`。设置 `G` 与 `P` 为对应的几何文件、原始 FP32 投影。

```sh
ncu --set full --kernel-name-base function -k back -c 1 ./build/bench_backproject 256 16 1
ncu --section MemoryWorkloadAnalysis --section Occupancy --kernel-name-base function -k pre -c 1 ./build/bench_filter 256 16 1
ncu --kernel-name-base function -k 'regex:^(angles|ramp|pre|spectrum|unpack|pairs|back)$' -c 7 --metrics sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread --page raw --csv ./build/cone-fdk "$G" "$P" 512 40 /dev/null
```

NCU 的 occupancy 只取每个自有 kernel 的首个 launch；它不代表 cuFFT 或全卷平均。向本机 NCU 请求 `dram__bytes_read.sum`、`dram__bytes_write.sum` 和 `dram__throughput.avg.pct_of_peak_sustained_elapsed` 后，导出的 raw 表没有这三列。故保留滤波、反投影的真实 DRAM 流量及峰值比例空缺；不拿 L2 代替。

## 端到端与 GPU 时间

```sh
nsys profile --sample=none --cpuctxsw=none --trace=cuda -o dense ./build/cone-fdk "$G" "$P" 512 40 build/volume.f32
nsys stats --report cuda_gpu_kern_sum dense.nsys-rep
```

性能表按单条 trace 内的 kernel 时长求和；表内的分母为同次跟踪中由父进程用单调钟包围 CLI 启动到退出的时长，不是 `nsys` 导出报告所耗时间。有效带宽 = 逻辑字节总数 / 阶段活跃时间，按十进制 GB/s。真实输入的一批有 `a` 角，探测器 `v×u`：滤波计 `a×(4vu+8(v(u+1)+2))` 字节；反投影计 `8n³`。另做无 profiler 的热缓存运行：预热一次再启五个独立进程，每次写并关闭同一普通输出文件、不 `fsync`。不能拿此数与旧 CLI 的持久化写入时间计算加速比。
