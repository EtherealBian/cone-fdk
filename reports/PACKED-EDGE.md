# 两遍 CUDA 调用完成一份带见证的边缘复核

多阈值提取现在可以从**已通过见证检查的复装作业**开始直接生成可批检、可离线打开的边缘作业。标称壁厚不重新估计：原始壁图逐字节复制；其余阈值按扫描合包，基准和复检各调用一次 CUDA 工具。旧程序与新程序并存，已有审查包和配方不随之改写。

| 在同一带噪父作业上，生成完整边缘作业 | 原路径中位数 | 合包路径中位数 | 原生子进程数 | 单个原生进程自有设备峰值（两边相同） |
|---|---:|---:|---:|---:|
| 三档（含标称），16 MiB 预算 | 690.11 ms | **445.13 ms** | 4 → 2 | 7,617,184 B |
| 九档（含标称），1 MiB 预算，强制分块 | 2,563.99 ms | **643.65 ms** | 16 → 2 | 1,048,504 B |

每组先预热，再交替执行六组 AB/BA；两种工况均 6/6 更快，每次输出壁图、无效状态与上下界均与旧路径一致。计时从读取就绪的父作业开始，到新的清单发布为止，包含来源校验、名义复制、原生进程启动、提取、拆包、状态计数和边缘判定。**不包含投影重建、姿态估计和后续区域分析**。同一 RTX 5060 Laptop、暖文件缓存、未锁频；不是整个 CT 流程快四倍。独立工具只计壁厚部分的数字另见[先前实验](WALL-PACK.md)。[六组原始配对、摘要和门槛](packed-edge-ab.json)

## 在另一个工位重跑

Windows 上从当前源码构建；以下使用同一已签入的模拟样例源指纹。自己的作业应从受信任的交付记录取得父清单和伴随程序的 SHA-256，不能仅因一份待验文件自己声明了某个摘要就信任它。

```powershell
# 如果还没有该批原始壳层，先从投影生成。使用新目录。
python scripts/validate_witness.py --region-review --keep build/packed-parents --reports build/packed-parents/reports
$p = (Get-Content reports/body-witness.json -Raw | ConvertFrom-Json).cases.'noisy-change'.manifest_sha256
$c = (Get-Content reports/wall-pack-build.json -Raw | ConvertFrom-Json).executable_sha256
python scripts/edge_review_packed.py build/packed-parents/noisy-change build/my-packed-edge --levels .018 .020 .022 --parent-sha256 $p --companion-sha256 $c
# 对本机刚生成、且信任的子作业做九种全局配对的区域复核；不复用原单档子作业的摘要。
$e = (Get-FileHash build/my-packed-edge/manifest.json -Algorithm SHA256).Hash.ToLowerInvariant()
python scripts/region_review.py build/my-packed-edge build/my-packed-regions --area-radius-mm 9.4 --review-sha256 $e
```

新入口保持来源文件锁，在核验旧 EXE、PTX、两份不同的投影来源、工装定位与体标记结果、标定、采样网格、壳层和标称壁图后才运行。不能从边缘作业反推见证通过。标称图还必须与父作业的原生减薄行程逐格一致；同数量但错位也会拒绝。新清单中 `backend`、两组 `packs`、各档来源、设备用量及进程数分别记录；包内单档摘要、输出字节计数、七类状态直方图均核对。有效但不完整的覆盖仍返回可复核结果（退出码 2），并保留不可测射线。坏来源、损坏元数据、第二个包失败或迟发现的标称不一致不发布子作业；输出目录必须未存在。

## 没有 CUDA 也可以核对本例

新程序的 10 个原始输出文件已与旧版样例相同，仓库只额外存 [6.8 KB 的新清单](packed-edge-example-manifest.json)，不重复存一份壁图。用 Python 3.11 或更新版本的标准库，即可核对新旧清单和每个文件，再还原这份**已有**作业。以下为本次发布的资格记录指纹，先从可信提交核对它：

```powershell
python scripts/restore_packed_example.py reports build/offline-packed --qualification-sha256 7f4e16fd0e057af4d83dd551f4348317ef6f78977951c1d5344f105c3a214f23
```

结果的边缘清单 SHA-256 应为 `ec33745f9f0141d388e7e1452af9cd1a394e031a8ff8fddd5b54e3d081cce6d2`。已有 Python 数值依赖时可将这个目录传给 `region_review.py` 重算九张区域图；不需要重新运行原生程序。还原出的作业不能作为新的实测或见证父作业。

## 新旧生产路径的边界验收

- 重新处理原样复扫、带噪减薄、故意使用无效边缘三组作业，共比较 28 个输出文件摘要。独立 CPU 从壳层重算 **236,160** 个径向壁厚记录，再检查 **324,720** 个配对单元。所有区域属性和标签均保持；新清单有不同后端和来源，只有在单独对照副本中替换父摘要后才要求区域清单逐字节一致。
- 工件滑移、缺少参考标记、只有夹具定位三项都在计算前拒绝，没有生成结果目录。故意损坏源、工具指纹、红区、状态直方图、标称行程，以及两组提取中间失败，也分别测试。
- [按作业的输出指纹、状态和独立回放](packed-edge-qualification.json) · [全回归与公开计时核对](packed-edge-regression.json)。`Run-Acceptance.ps1 -PackedCampaign -SpecimenCampaign` 会从投影重新生成父作业；对照原记录而非新鲜生成的摘要给自己的结果打分。所有新壁厚案例为仿真，实测核桃只做重建回归。

隔离工作树上双重原生构建后，非精简全流程的 400 项测试全部通过。加入离线还原与计时记录的破坏性反例后，在同一工作树重新跑完整测试集：417 项通过，无跳过。
