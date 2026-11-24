# 空间模型与拆孔实验：哪些改动能保留？

**两种方法均未晋级。** 候选 F 用原来的 24 张校准切片训练小型卷积网络。它在试件 3 多找到了一个参考孔，但在试件 2、5 上把体积误差尾部拉大了。拆孔是另一个独立实验，使用候选 E；它在新区域误拆一个原本匹配的孔，没找回真正被合并的孔。两项都没有使用保留试件 4、6，也没有生成最终确认清单。

## 正式开发区域：同一批原图，E → F

范围 z=[400,528)。F 的[训练和评分规则](EXPERIMENT-F.md)先于运行发布；[三件预测摘要](https://github.com/EtherealBian/cone-fdk/commit/92289a8)在本轮评分读标注前公开。开发区域在以前的候选上用过，不能称为盲测。P95 是每件所有匹配孔的**绝对相对体积误差**的第 95 百分位；上限 15%，中位数上限 5%。

| 试件 | Dice F | F 匹配 / 预测 / 参考 | 体积中位数 E → F | 体积 P95 E → F | 通过门槛 E → F |
|---|---:|---:|---:|---:|---:|
| [2](evidence/candidate-f/sample2.json) | 0.94481 | 69 / 72 / 73 | 4.40% → 4.41% | 20.97% → 23.36% | 10 → 9 / 11 |
| [3](evidence/candidate-f/sample3.json) | 0.96924 | 25 / 33 / 27 | 7.45% → 11.64% | 28.16% → 27.66% | 6 → 5 / 11 |
| [5](evidence/candidate-f/sample5.json) | 0.97934 | 11 / 13 / 12 | 6.59% → 7.46% | 24.76% → 32.34% | 7 → 6 / 11 |

* 只看两种方法都找到的孔，2 号 66 个的 P95 从 15.98% 变为 23.85%；3 号 24 个从 28.16% 变为 26.36%；5 号 11 个从 24.76% 变为 32.34%。[逐个参考孔](evidence/candidate-f/sample2-paired-e-f.csv)包括新增、丢失以及两者都漏掉的参考孔。5 号唯一共同匹配的 64–127 体素小孔确实从 28.36% 降到 7.46%，但不能用这一个孔代表整件。
* 每件 8,388,608 个正式区域体素上，自写 CUDA 卷积与 CPU PyTorch 的二值判断逐体素一致；三件最大 logit 差分别为 0.00001145、0.00001145、0.00002289（向上取位）。输入特征共用 CUDA 去噪；这不是两套独立去噪实现。按同一掩膜计量，188、223、79 个连通体逐行通过 SciPy 对照，8/37 层分块的 CSV 字节相同。独立复核从认证源档检查全部参考掩膜与阅片图像。**实现一致不等于与数字参考一致。**

## 新区域：两个高置信核心是否真意味着两个孔？

[事先发布的条件](SPLIT-PILOT.md)与[预测封存提交](https://github.com/EtherealBian/cone-fdk/commit/70b6d2e)；只看 z=[552,680)，不替代正式评估。新区域 2/3/5 的实际粘连事件分别为 **1/0/0**，成功拆回为 **0/0/0**。试件 2 有两个不同的反例：

* **漏报粘连。** 原生 ID `442336` 覆盖参考孔 33 和 60，但只出现一个满足条件的核心。算法没有操作。
* **误拆单孔。** 原生 ID `7059649` 匹配参考孔 172（943 体素）。去掉两个桥接体素后，904 个体素被分成 817 和 85；小块没有对应的参考孔。合格预测从 79 增至 80，匹配仍是 78。该件全匹配 P95 没变（17.35%），**只看这个百分位会漏掉误拆**。3、5 号没有动作，不能当作成功拆分的证据。

[所有母体、切割体素和判定](evidence/split-pilot)保留。自动拆孔不进入正式算法；第二个高置信核心也不能作为独立验证过的粘连警报。

## 校准对照：是否应排除均值或方差不同的邻块？

在事先隔开的校准窗口，加入两种局部统计门槛后，三个试件的 Dice 都降低了：2 号 0.94405 → 0.78699、3 号 0.95829 → 0.91024、5 号 0.98119 → 0.95421。此次固定筛选不用在正式区域。[事先公布的条件](PATCH-STATS-PILOT.md) · [完整混淆计数](evidence/patch-stats.json)。

## 独立复算

下载 [三份离线包](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-4)，按 [SHA-256](evidence/candidate-f/release-SHA256SUMS.txt) 核对后解压到同一目录。`formal/sample*-cases.html` 可定位误差，`formal/sample*-review.html` 可逐层查看。数据来源：Kim 等，NIST，DOI 10.18434/M32162。数字参考不是物理量具。

```powershell
# 在 measurement-evidence-4 源码目录，沿用 README 中指定的 Python 与原生构建环境。
python pores/audit.py reviews/formal --development 2 3 5 --cache $cache
# 此次应返回 1 并逐件列出失败指标；哈希/数值不一致则抛错。
python scripts/pore_split_replay.py reviews/split --cache $cache
python scripts/pore_pairs_ef.py pores/evidence/candidate-e reviews/formal --check
# 如需重新做 CUDA 两种分块、重建全部阅片器及评分：
python scripts/pore_spatial_score.py --baseline reviews/predictions --cache $cache --out build/F-rescore
# 如需从原始数据重新训练，单设一个目录，避免更改旧候选的 Python 环境：
python -m pip install --target build/torch-lab -r pores/spatial-requirements.lock
$env:PYTHONPATH=(Resolve-Path build/torch-lab).Path
python scripts/pore_spatial_predict.py --cache $cache --out build/F-rerun
```

完整开发流程使用自写 CUDA 去噪、卷积及连通计量；F 校准训练用 CPU，研究性最小割和独立复核也在 CPU。原稳定重建版仍为 `v0.1.0`；验收遵循 [FINAL-GATE](FINAL-GATE.md)。
