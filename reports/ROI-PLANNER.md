# Fewer projection reads under a tight GPU budget · 低显存预算下的多选区调度

On one Windows GPU, reconstructing the same six boxes from 720 measured walnut projections took **0.590 → 0.304 s** at a **48 MiB** device-buffer budget. The new opt-in policy read the projections **once instead of three times**. Every measured run produced the same pack as the [published parent-crop comparison](MULTI-ROI.md).

## 为什么小一点的批次反而有用？

一次多选区重建需要同时放下两类数据：本批角度的投影和滤波工作区，以及尚未累加完所有角度的选区体素。原策略 `batch-first` 先争取较大的角度批次。在本例中，16 个角度放得下，但留给六个选区的空间不够；换下一组选区时又要从头读取、校验和滤波 720 幅投影。改为每批 8 个角度，六个选区可以同时常驻，滤好的投影一次用完。

| 同一程序、同一 48 MiB 预算、512³ 父网格、40 mm 视野 | 默认 `batch-first` | `min-passes` |
|---|---:|---:|
| 实际批次（请求上限均为 16） | 16 | 8 |
| 完整投影遍数 | 3 | 1 |
| 应用读入字节（不是 SSD 物理流量） | 716,083,200 | 238,694,400 |
| 滤波批次 | 135 | 90 |
| 反投影内核启动次数 | 315 | 540 |
| 引擎自有设备缓冲峰值，不含驱动与上下文 | 47.95 MiB | 35.38 MiB |
| 整进程耗时中位数，含选计划、读入、计算、写出及替换 | 0.590 s | **0.304 s** |

两种策略各预热一次，再做六组交替 AB/BA；不清系统文件缓存，不锁频。新策略 **6/6 组更快，中位耗时减少 48.5%**；门槛是至少 5/6 胜出且中位数至少降低 10%。反投影启动次数实际上增加了：这个策略减少的是整套投影的重复处理，不是最小化内核启动次数，也不是为所有输入预测最快耗时。

## 哪些预算会改变调度？

同一输入、请求上限均为 16。除 48 MiB 外，每档每策略各运行一次，只验证调度和结果，不据此给出加速比。

| 引擎设备缓冲预算 | 默认：实际批次 / 遍数 | 新策略：实际批次 / 遍数 |
|---:|---:|---:|
| 24 MiB | 8 / 6 | 2 / 1 |
| 32 MiB | 8 / 2 | 4 / 1 |
| 40 MiB | 8 / 1 | 8 / 1 |
| 48 MiB | 16 / 3 | 8 / 1 |
| 56 MiB | 16 / 2 | 8 / 1 |
| 64 MiB | 16 / 1 | 16 / 1 |
| 128 MiB | 16 / 1 | 16 / 1 |

每次都对照同一个已发布的整包 SHA-256：`bf555f872da15e51ddb9501dfd503e98873bbaa414f40e2ff11992a98545ef1f`，包括索引、坐标顺序和所有 FP32 体素。各次设备缓冲都在预算内且通过边界哨兵检查。这验证了已测数据的数值没有变化，不是新的物理测量精度，也不保证跨 GPU、cuFFT 版本或任意输入都逐字节相同。数据：Alexander Meaney / 赫尔辛基大学，CC BY 4.0，[来源及处理说明](../NOTICE.md)。

## 使用与复核

```powershell
# 当前源码构建；data/full-corrected 是已准备好的 720 角投影前缀
$py = '.\.venv\Scripts\python.exe'
& $py scripts/reconstruct.py data/full-corrected build/six.croi --n 512 --fov 40 --budget 48 --batch 16 --roi-list fixtures/walnut-six-rois.txt --planner min-passes
& $py scripts/validate_planner.py --scan data/full-corrected --scratch build
```

- **默认不变。** `min-passes` 只对 `--roi-list` 启用；省略参数或显式 `batch-first` 都沿用原有选法。清单顺序、拆层规则及包格式不变；原 v0.1.0 二进制不支持此命令。
- 从 `min(请求批次, 角数)` 开始依次整除 2；只评估一张最大 XY 平面、滤波缓冲和 **cuFFT 实际返回的工作区**都能放下的候选。用执行时的同一选区切分算法统计遍数，遍数相同保留更大批次；找到单遍即停止。这里的最少遍数是指**这个候选集合**，不遍历每个整数，不重排选区，也不试跑重建来选计划。
- 原生 JSON 里的 `requested_batch` 与 `batch` 区分请求值和执行值；`feasible_plans_scored` 列出实际评估的可行候选。`effective_budget_bytes` 还考虑了当时可用显存；`projection_passes`、`projection_bytes_read` 和 `filter_batches` 是执行时计数。
- [每一轮的原始耗时、指纹、预算与读入量](roi-planner-walnut.json) · [验收脚本](../scripts/validate_planner.py) · [80 项回归记录](roi-planner-tests.xml)。在另一组 37 角解析投影上，测试分别强制实际运行候选批次，核对预测遍数、并列时选择、尾批、输出字节和拒绝错误输入时保留已有文件。

## 完整验收复跑

同一可执行文件还在隔离目录通过了完整 `Run-Acceptance.ps1 -SkipBuild`。从 8 个已校验的原始归档重新处理 720 幅投影，默认路径生成的 512³ 完整体哈希与已发布参考一致，对 ASTRA 全体素相对 L2 为 **0.02899%**。1296³ 解析试件输出的 8,707,129,344 字节也与历史参考哈希相同，另外用 ASTRA 检查了五个切面。这次独立流程自带的六组 48 MiB 配对为 **0.554 → 0.291 秒，6/6 更快**；上面的主表仍使用第一次正式验收的六组数据。另两次源码构建得到同一套 EXE/PTX，和实际受测文件哈希一致。[完整验收摘要、二次配对原始耗时和构建指纹](roi-planner-full-check.json)。
