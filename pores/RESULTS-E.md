# 候选 E：小孔边界的逐件复核

[评分前登记的方法](EXPERIMENT-E.md)使用新的 CUDA 核函数，分别用最相似的约 8、32、96 个图像块估计灰度；校准时给小孔边界内外两侧加权。这里检验 NIST 钴铬 XCT 与**数字分割**的一致性，每件使用自己的少量标注校准。不是物理尺寸认证。

## 原定 11 项门槛：逐件计算，不取三件平均

| 试件 | 方法 | Dice | 孔隙率偏差（百分点） | 匹配 / 合格预测 / 合格参考 | 逐孔体积绝对误差中位 / P95 | 通过项 |
|---|---|---:|---:|---:|---:|---:|
| [2](evidence/candidate-d/sample2.json) | D | 0.94767 | +0.01038 | 66 / 68 / 73 | 4.08% / 20.68% | 9/11 |
| [2](evidence/candidate-e/sample2.json) | E | 0.95487 | -0.00105 | 67 / 71 / 73 | 4.40% / 20.97% | 10/11 |
| [3](evidence/candidate-d/sample3.json) | D | 0.97665 | +0.01354 | 25 / 29 / 27 | 6.08% / 33.16% | 7/11 |
| [3](evidence/candidate-e/sample3.json) | E | 0.97755 | -0.01704 | 24 / 30 / 27 | 7.45% / 28.16% | 6/11 |
| [5](evidence/candidate-d/sample5.json) | D | 0.98345 | -0.00625 | 10 / 15 / 12 | 7.65% / 18.73% | 6/11 |
| [5](evidence/candidate-e/sample5.json) | E | 0.98404 | -0.01454 | 11 / 15 / 12 | 6.59% / 24.76% | 7/11 |

逐孔误差以各参考孔自身的体积作分母。P95 上限为 15%。未匹配孔不参与百分位，但记入覆盖率；完整失败项如下：

- 试件 2：`p95_abs_volume_error_pct`。
- 试件 3：`object_prediction`、`object_reference`、`median_abs_volume_error_pct`、`p95_abs_volume_error_pct`、`median_abs_diameter_error_pct`。
- 试件 5：`object_prediction`、`median_abs_volume_error_pct`、`p95_abs_volume_error_pct`、`median_abs_diameter_error_pct`。

## 同一批孔有没有测得更准？

| 试件 / 全量 CSV | 共同匹配数 | D 同孔 P95 | E 同孔 P95 | 误差变小 / 相同 / 变大 | 新匹配参考 ID | 丢失参考 ID | 两者均未匹配的参考 ID |
|---|---:|---:|---:|---:|---|---|---|
| [2](evidence/candidate-e/sample2-paired-d-e.csv) | 66 | 20.68% | 15.98% | 39 / 4 / 23 | 158 | — | 29, 38, 50, 58, 70, 93 |
| [3](evidence/candidate-e/sample3-paired-d-e.csv) | 24 | 33.55% | 28.16% | 11 / 0 / 13 | — | 197 | 177, 199 |
| [5](evidence/candidate-e/sample5-paired-d-e.csv) | 10 | 18.73% | 18.61% | 5 / 0 / 5 | 67 | — | 21 |

CSV 包含每个合格参考孔，列出预测体积减参考体积的体素差，以及仅预测、仅参考的体素数。未匹配的格子留空，不用零误差填补。

- **试件 3，参考孔 197：**参考 77 体素，D 测得 71；E 与它最重叠的连通体只有 59 体素。虽然 IoU 为 0.639，它仍低于 64 体素的资格线。这个孔记为漏检，不进入 E 的误差百分位。
- **试件 5，参考孔 67：**参考 67 体素，D 的最佳重叠 IoU 为 0.476，未达到 0.5。E 新匹配到它，却测得 86 体素（高估 28.36%）。所以共同孔的 P95 几乎没变，把新增孔纳入正式计分后，整件 P95 反而更高。

### 放大看 64–127 体素的小孔

| 试件 | 合格参考数 | D 匹配 | E 匹配 | 共同匹配 | D 共同孔 P95 | E 共同孔 P95 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 20 | 14 | 15 | 14 | 24.90% | 28.78% |
| 3 | 13 | 11 | 10 | 10 | 38.75% | 31.99% |
| 5 | 2 | 0 | 1 | 0 | — | — |

“—”表示没有共同样本；很小的样本数不支持稳定的尾部分布估计。完整分段表和物理单位换算见各件 `*-size-bands.csv`。

## 哪些孔值得人工再看？

[事先登记的复核规则](DISAGREEMENT.md)只比较 C、D、E 在同一位置的预测：找不到合格的同伴、IoU 不足 0.5，或任一同伴体积与 E 相差超过 E 的 15%，都标记出来。随后才用参考掩膜检查标记的效果。下表中“有问题”指 E 未匹配，或匹配但体积误差超过参考体积的 15%。

| 试件 / 标记 CSV | 标记且有问题 | 标记但无此问题 | 未标记却有问题 | 未标记且无此问题 |
|---|---:|---:|---:|---:|
| [2](evidence/candidate-e/sample2-review-queue.csv) | 3 | 1 | 7 | 60 |
| [3](evidence/candidate-e/sample3-review-queue.csv) | 5 | 3 | 5 | 17 |
| [5](evidence/candidate-e/sample5-review-queue.csv) | 2 | 0 | 5 | 8 |

未标记不能视为通过，多个方法可能犯同一个错误。E 完全漏掉的参考孔也不可能出现在按预测生成的队列中；完整验收始终包括这些漏检。

## 复现与离线复核

- [三个完整复核包与 SHA-256](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-3)。解压后在浏览器打开 `*-cases.html` 按误差找孔，或在 `*-review.html` 选择数字 ID 核对原始切片。`*-review-queue.html` 还可以在同一坐标并排查看原图及 C/D/E 三种预测。
- 预测源码 `e50a3e8`；运行 `python pores/candidate.py --model E --specimens 2 3 5 --cache <archive-directory>`。包中的 CUDA 组件表与两种分块的字节一致；独立 `pores/audit.py` 可根据原始源档、复核掩膜和组件表重算门槛。
- 从完整 C/D/E 复核目录运行 `scripts/pore_review_queue.py`；用 `scripts/pore_pairs_de.py` 从 D/E 的逐件匹配表和交集表重建配对 CSV。`scripts/pore_report_e.py --check` 检查本页与公开证据是否一致。
- [校准网格对照](GRID-PILOT.md) · [切片方向对照](PLANE-PILOT.md) · [上轮 D](RESULTS-D.md) · [最终验收标准](FINAL-GATE.md)。开发未通过，预留的 4、6 未评分；稳定核桃重建版仍为 v0.1.0。
