# 多档壁厚：不重复上传同一块壳层

给同一组圆柱壳层选八个边缘阈值，原来的方法启动八次原生进程并传八遍剖面。新增的 `wall-pack.exe` 每个块只从磁盘读取、上传一次；复用同一个 `radial_wall` PTX 和输出缓冲逐档计算，落盘为阈值优先的 32 字节壁厚记录。旧 EXE、旧 PTX、无效状态和标称壁厚定义不变。

| Windows GPU，壁厚阶段 | 单档进程中位数 | 合打包中位数 | atlas 实读（旧 → 新） | 两边峰值自有设备缓冲 |
|---|---:|---:|---:|---:|
| 2 个非标称档，16 MiB 预算 | 296.51 ms | 155.91 ms | 14,287,680 → 7,143,840 B | 7,617,184 B |
| 8 档，1 MiB 预算，8 块/档 | 1219.11 ms | 179.91 ms | 57,150,720 → 7,143,840 B | 1,048,504 B |

两种工况各交替做 6 组 AB/BA，每组新路径都更快，各档逐字节和七类状态计数一致。[完整试次、输入及可执行文件摘要](wall-pack-ab.json) · [伴随程序双重构建](wall-pack-build.json)。计时包含进程启动、原生校验、写盘；新路径亦计入拆包、逐档摘要与刷新，旧路径额外生成扇区摘要。未锁频、未清系统文件缓存，指标不含重建、配准或端到端检验；设备缓冲不含驱动保留。输入是模拟复装作业留存的 360×41×121 采样壳层，不是实物标定。

```powershell
# 在仓库根目录按 README 编译。仅使用新的输出目录。
$p = 'build/witness-final-replay/noisy-change'
python scripts/wall_pack.py "$p/inspected/shells.f32" "$p/inspected/scanner.grid" build/wall-pack-demo --levels .018 .022 --budget 1 --unpack
# 独立复跑带源摘要的配对计时；不与其它 GPU 测试同时跑。
python scripts/benchmark_wall_pack.py $p build/wall-pack-trials build/wall-pack-trials.json --parent-sha256 20ce077eb45c583d3620d40b814cddd0317c0107d0b17bfff1eb3d6877e9f2e3
```

独立工具最多接收 16 档严格升序、有限、正阈值。每档保持原始 NaN 与拒测状态。输出清单最后发布；只出现中间文件的目录不算完成。单独运行本工具不会审定夹具见证或生成减薄结论，受见证保护的历史 `edge_review.py` 不受影响。

需要从受见证保护的父作业生成完整的新边缘作业时，请改用[带来源检查的合包入口](PACKED-EDGE.md)。
