# 灰度校正后的逐孔结果，以及三种方法都判错的位置

**候选 G 未通过，保留试件 4、6 仍封存。** F 直接学孔的分数；G 学去噪灰度的有限修正，再用固定的局部阈值判孔。两者使用相同的 24 张校准切片。[G 实验规则](EXPERIMENT-G.md)和[三件预测摘要](https://github.com/EtherealBian/cone-fdk/commit/043a7c7)先于 G 的正式评分公开。开发区曾用于其他候选，不是盲测。

| 正式试件 | G 匹配 / 合格预测 / 合格参考 | G Dice | G 体积误差中位数 | 体积误差 P95：E → F → G | G 通过项 |
|---|---:|---:|---:|---:|---:|
| [2](evidence/candidate-g/sample2.json) | 69 / 72 / 73 | 0.95008 | 3.33% | 20.97% → 23.36% → 17.04% | 10/11 |
| [3](evidence/candidate-g/sample3.json) | 19 / 28 / 27 | 0.96906 | 11.49% | 28.16% → 27.66% → 43.25% | 5/11 |
| [5](evidence/candidate-g/sample5.json) | 10 / 16 / 12 | 0.97749 | 6.51% | 24.76% → 32.34% → 23.86% | 5/11 |

P95 按每件的所有匹配孔计算，上限 15%。2 号只有这一项未过。3 号比 F 少匹配了 6 个参考孔，同一批 19 个共同孔的 P95 也从 26.99% 升到 43.25%。5 号虽然 P95 下降，却丢了参考孔 26、67，另找回孔 21。所有新增、丢失及两边都漏掉的 ID 均在 [E/G、F/G 配对表](evidence/candidate-g)。不能按试件拼接最好的结果。

每件 8,388,608 个体素。原生 8 层和 37 层分块的对象 CSV 字节相同，180、272、107 个连通体全部逐行对齐 SciPy。除了从认证源档复核参考掩膜和阅片器，还[重新从原始灰度和存档权重推理](evidence/candidate-g/raw-weight-replay.json)。第二条路径共用 CUDA 去噪，以独立组装的 CPU 卷积和阈值运算复核；2/3/5 的最终二值差异均为 0。3 号含边界上下文的 8 位中间图有 **4 处**不同，决策边距的最大数值差为 1；其余两件试件这两项均为 0。[F 的同类权重复算](evidence/candidate-f/raw-weight-replay.json)也已补齐。

## 若能修对每个分歧体素，剩下什么误差？

[事先规定的诊断](AGREEMENT-PILOT.md)只在已公开的开发区进行。以 E/F/G 三个二值预测做多数表决；另一列固定三者一致的体素，在分歧处直接用参考取值。后者**读取答案**，只能诊断，不能作部署算法或验收成绩。

| 试件 | 分歧体素 | 三者都误报 / 都漏报的体素 | 多数表决 P95 | 分歧处全改对：Dice / P95 | 仍超 15% 的匹配参考孔 ID |
|---|---:|---:|---:|---:|---|
| 2 | 7,981 | 1,733 / 1,997 | 19.46% | 0.97495 / 15.89% | 44, 85, 93, 157, 158 |
| 3 | 92,564 | 19,719 / 19,215 | 28.34% | 0.98698 / 15.96% | 89, 127 |
| 5 | 44,803 | 6,588 / 9,030 | 26.52% | 0.99163 / 22.22% | 92, 105 |

条件内逐体素错误已经最少，但每件的逐孔 P95 都仍然高于门槛。这**不是**逐孔体积 P95 的数学下界：边界上多算和少算的体素可能抵消，连通性也会改变匹配。结果表明需要检查几种方法共享的输入或判定假设，而非把重复投票当成独立证据。[八种投票组合、全部参考孔和原生计量表](evidence/agreement-efg)。

## 复算

从[研究发布页](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-5)下载三个 ZIP，对照 [SHA-256](evidence/candidate-g/release-SHA256SUMS.txt)，解压到同一目录。每件的 `formal/sample*-cases.html` 能按误差定位孔；`formal/sample*-review.html` 可看原始切片、最终掩膜和数字参考。源数据为 NIST Kim 等，DOI 10.18434/M32162。这里的误差针对数字分割，不是物理计量认证。

```powershell
# measurement-evidence-5 源码；按 pores/README.md 构建原生程序、PTX 及锁定的 Python/Torch 环境。
python pores/audit.py reviews/formal --development 2 3 5 --cache $cache
# 验收失败会正常返回 1；证据、来源或数值不一致会抛错。
python scripts/pore_field_replay.py --model G --predictions reviews/predictions --formal reviews/formal --cache $cache --out replay-new.json
python scripts/pore_pairs_ef.py pores/evidence/candidate-f reviews/formal --stages f g --check
# 将 E、F、G 三个完整离线 formal/review 集目录分别命名为 build/candidate-E、-F、-G：
python scripts/pore_agreement.py --root build --cache $cache --out build/agreement-new
```
