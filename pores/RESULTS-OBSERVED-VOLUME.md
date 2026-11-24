# 24 层校准图里，到底有多少个孔能教算法判断体积偏差？

在每件原来的 24 层校准图里，先用**没有用该件训练**的检测器画孔，再与该件的源参考配对。只问「实际看得见的三维部分量大了还是量小了」，不从截面猜测切掉的体积，也不把一个孔的多张相邻截面当成多个孔。

|试件|符合条件的预测 / 参考|完整配对|仅沿 z 被切的配对|合计低估 / 高估 / 相等|可见体积误差超 15%|最低方向支持|
|---|---:|---:|---:|---:|---:|---|
|2|29 / 27|13|13|8 / 18 / 0|4 / 26|满足|
|3|16 / 13|0|12|10 / 2 / 0|4 / 12|满足|
|5|5 / 5|0|5|4 / 1 / 0|1 / 5|**不满足**|

这项检查事前要求：每件至少 5 个不同的配对结构，至少各见到两次低估和高估。放宽到只比较截获的体积，3 号确实从零个完整配对得到 12 个可用例子；5 号的 5 个例子全部截断，只有一次高估。在这里，「看到过一次」不足以满足预先定下的支持条件。以上数量是对当前留出检测器和匹配条件而言，不是在宣称 5 号原始扫描只有五个孔，也不能据此推算其他试件的误差或理论精度。

**工程决策：不直接用这组稀少的三维体积误差训练新选择器，再把它包装成通用模型。** 下一个数据方向是从独立开发来源获取真正的连续三维原图和配准分割，在训练前核对体素间距、标注定义、完整孔及两种误差方向的数量。如果只有独立二维切片，不把它拼成三维量体积。外部样本只能用作独立研究/训练资料，不能替代 NIST 原定的正式开发与封存验收。可以查阅 [XCTPore 作者的原始数据说明（固定版本）](https://github.com/BismaMutiargo/XCTPore/blob/aa8e71b8ed1b2c63ba599da2d059b5323710d752/README.md)；本次未下载或纳入该数据，也未把其数组的第一维当作已证实连续的扫描深度。

**复核。** 预测与参考均须在观测区内有至少 64 个体素，不触 XY 边，在 z 的两端有完全相同的触边标志。逐孔 6 连通、IoU 至少 1/2、精确分数排序和一对一匹配；查 [全部配对及未配对记录](evidence/observed-volume-support/support.json)。Native CUDA 的预测/参考组件表分别用 8 与 17 层分块运行，与 SciPy 核对每个组件的体积、种子位置、六面触边标志等。独立重复整个过程，六张 CSV 和一份 JSON 逐字节一致，见[重放摘要](evidence/observed-volume-support/replay.json)。这是训练信号清点，不是新的测量成绩，不涉及正式区域、另一深度或保留件。

[事前协议](OBSERVED-VOLUME-SUPPORT.md) · [为何改查三维校准支持](RESULTS-CROSSFIT-GATE.md) · [正式验收规则](FINAL-GATE.md)。

```powershell
python scripts/pore_observed_volume.py --cache $cache --out build/new-observed-a
python scripts/pore_observed_volume.py --cache $cache --out build/new-observed-b
python scripts/pore_observed_replay.py --first build/new-observed-a --second build/new-observed-b --out build/new-observed-replay.json
```
