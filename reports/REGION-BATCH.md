# 离线批检：中断后从哪一件继续？

一批可以同时包含完整复核、含不可测区域的复核，以及上游因工装或见证失败而拒测的作业。每件有自己的来源清单、网格、阈值和结果文件；不拼接不同试件的坐标，不跨件复用姿态，也不把“拒测”折算成零缺陷。这里的计数单位是**作业条目**，不是独立实物数量；同一试件重扫并不会增加实物样本量。

## 最小可运行例子

先按[区域复核](REGION-REVIEW.md)将公开包解压至 `build/region-edge-example`。以下两次使用同一配方，第二次会校验当前来源及结果摘要，但不重复计算连通区域。

```powershell
python scripts/region_batch.py fixtures/region-offline-batch.json build/region-batch
python scripts/region_batch.py fixtures/region-offline-batch.json build/region-batch --resume
```

[单件配方](../fixtures/region-offline-batch.json)与[五条混合配方](../fixtures/region-campaign-batch.json)均可直接查看。`job` 相对配方所在目录解析，也可用本机绝对路径。每个条目都需要来自可信记录的 `manifest_sha256`；`edge` 条目还需指定面积参考半径 `area_radius_mm`，并且半径必须在当前采样壳层内。`witness-refusal` 只转述经过摘要校验的上游拒测记录和原因，不声称重新运行了上游定位。输入目录和输出目录不能互相包含。每批 1–1024 个条目；ID 不区分大小写且不得重复。

## 结果与恢复语义

- `manifest.json` 是**本次批检结束**的索引，记录每条来源、状态和可打开的 `results/<sha256>.json` 路径。`complete_coverage` 表示每组设置都能逐格测量，不表示缺陷合格。`incomplete_coverage` 和 `upstream_refused` 均保留为独立状态。输入错误记为 `input_error`，没有结果指针，不阻止其他件完成。命令退出 0 表示仅完整覆盖；2 表示存在拒测或不完整覆盖；1 表示输入错误或批处理失败。
- 单件报告以内容摘要寻址，先落盘，再提交到 SQLite 本地日志。批处理一次只允许一个写者。中断后保留已提交的件；恢复时逐件校验原始数据和已有结果，再跳过连通计算。恢复验证期间不展示旧的完工索引。报错输入恢复原字节后可以续跑；损坏的检查点不自动重写。开始新批次必须使用新目录。
- 配方原始字节、批处理与测量脚本摘要以及 Python、NumPy、SciPy 版本绑定到当前日志。改配方或相关版本须另开目录。SHA-256 不是签名：需要信任来源配方、依赖和本地输出目录，不应信任别人修改过的日志数据库。结果文件是普通 JSON，不需要 SQLite 工具即可复核。

## 已验收的混合批

在保留的模拟作业上，原样复扫与带噪声内凹两条覆盖完整，故意设错边缘阈值的一条覆盖不全，工装滑移和基准见证缺失两条拒测。首轮生成五条结果，恢复时复用五条、重算零条，批索引逐字节相同。验收还会在第一件提交后打断运行、篡改一张源壁图、损坏缓存结果、改配方和争抢同一写者锁，分别检查能否恢复以及是否错误地沿用旧结论。

[混合批清单摘要、两个拒测来源和五个可核对结果](region-batch-qualification.json)。复跑命令：`python scripts/validate_region_batch.py fixtures/region-campaign-batch.json build/batch-check reports build/batch-check-evidence.json`。
