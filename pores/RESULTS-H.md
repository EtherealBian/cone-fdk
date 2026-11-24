# 候选 H：共享权重能不能把不同试件的孔量准？

**没有通过。** 同一套权重在开发试件 2、3、5 上分别通过 7、4、5 项，要求是每件 11/11。保留试件 4、6 未启用。此处比较的是 NIST 数字分割，不能据此认证实物孔径。

H 在每件原有的 24 张校准切片上联合训练：除了逐体素监督，还要求一个完整小孔预测的软体积接近其参考体积。满足训练取样条件（16–511 体素、足够边界余量）的孔分别只有 **14、3、1** 个。5 号的 150 次完整孔抽样都来自同一个 46 体素孔；不能将抽样次数当成独立样本量。[训练协议](EXPERIMENT-H.md)；[评分前公开的三件预测和唯一权重摘要](https://github.com/EtherealBian/cone-fdk/blob/e0e36f0/pores/evidence/candidate-h/predictions.json)。这是开发预测封存，不是保留试件验收清单。

| 开发试件 | 匹配 / 合格预测 / 合格参考 | Dice | 匹配孔体积误差中位数 | 体积误差 P95（目标 ≤15%） | 通过项 |
|---|---:|---:|---:|---:|---:|
| [2](evidence/candidate-h/sample2.json) | 64 / 71 / 73 | 0.93085 | 4.24% | 22.61% | 7/11 |
| [3](evidence/candidate-h/sample3.json) | 17 / 26 / 27 | 0.96419 | 17.68% | 39.65% | 4/11 |
| [5](evidence/candidate-h/sample5.json) | 7 / 10 / 12 | 0.97525 | 11.57% | 18.96% | 5/11 |

[全部失败项](evidence/candidate-h/summary.json)。2 号虽通过两项孔隙率误差门槛，却未通过 Dice、精确率、参考孔覆盖率和体积误差 P95。3、5 号的像素级 Dice 均达标，预测和参考两侧的对象覆盖率却都没有达标。

### 比较同一批孔，结论会反过来

| 试件 | G、H 都匹配的参考孔 | 共同孔体积误差 P95：G → H | H 新匹配 / 丢失 |
|---|---:|---:|---:|
| [2](evidence/candidate-h/sample2-paired-g-h.csv) | 62 | 16.12% → 22.69% | 2 / 7 |
| [3](evidence/candidate-h/sample3-paired-g-h.csv) | 17 | 43.65% → 39.65% | 0 / 2 |
| [5](evidence/candidate-h/sample5-paired-g-h.csv) | 7 | 9.39% → 18.96% | 0 / 3 |

5 号如果只看各自匹配的全部孔，G → H 从 23.86% 降至 18.96%。但 H 丢了参考孔 **21、92、105**。孔 21 的参考大小为 66 体素，G 匹配到 64 体素；H 最佳重叠对象只有 49 体素，IoU 虽为 0.643，却低于 64 体素的大小门槛。孔 92 对应的 72 体素预测达到大小门槛，但 IoU 仅 0.4965035，不能四舍五入当作 0.5。孔 105（137 体素）只与一个 7 体素对象有最佳重叠。阅片器可按原始坐标定位这三类误差；E/H、F/H 的参考 ID 配对也在包内。

### 可复核部分

每件正式 ROI 含 8,388,608 体素。8 和 37 层分块得到字节一致的对象表；2/3/5 分别有 203、341、94 行经 SciPy 逐对象核对。独立进程从原始档读取图像，用存档共享权重在 CPU 重算卷积，每件最终二值判定均为零差异；浮点输出最大绝对差约为 2.29e−5、1.53e−5、1.91e−5。[逐件重放](evidence/candidate-h/raw-weight-replay.json)。该路径与原生路径共用 CUDA 去噪器，验证的是后续推理和逐层掩膜，不是对去噪实现的独立验证。

下载[三个离线包](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-6)，核对 [SHA-256](evidence/candidate-h/release-SHA256SUMS.txt)，解压到同一目录。打开 `formal/sample5-cases.html` 定位漏配，或在 `formal/sample5-review.html` 看原始层与掩膜。使用此标签的源代码、锁定环境和 NIST 源档（DOI 10.18434/M32162）：

```powershell
python pores/audit.py reviews/formal --development 2 3 5 --cache $cache
# 正确重放失败的开发候选：返回 1，并列出每件未通过的项目。
python scripts/pore_field_replay.py --model H --predictions reviews/predictions --formal reviews/formal --cache $cache --out h-replay.json
python scripts/pore_pairs_ef.py pores/evidence/candidate-g reviews/formal --stages g h --check
```
