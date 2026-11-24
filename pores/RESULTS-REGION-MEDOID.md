# 区域投票能把已发表的预测组合成合格测量吗？

**不能。** 在开发迁移区，算法不知道参考答案，在每个六连通区域选出与其他六组预测总体最一致的一组。与原 census 相比，2 号找回一个孔，但体积误差变大；3 号体积误差有所下降，未达到事前规定的 2 个百分点；5 号没有增加匹配。七组预测固定、每个区域只能整组选一次时，5 号即使事后知道答案，也不可能达到参考孔覆盖线。这不代表其他原图算法也无法达到。

### 可运行的预测器：分母、体积误差和得失

下表三列都按「匹配 / 合格预测 / 合格参考；全部匹配孔绝对相对体积误差 P95」显示。体素是否被七组预测中的至少四组判成孔，是“多数”对照。区域选择的打分是各供选掩膜之间 Jaccard 的精确有理数和，不含参考 ID、体积或强度阈值。包含所有边缘和空选择，不删麻烦的孔。

|试件|原 census|体素多数|按区域选整组|区域法 Dice；迁移区通过项|
|---|---|---|---|---|
|[2](evidence/region-medoid/sample2.json)|42/44/49；14.89%|43/45/49；20.89%|43/45/49；19.94%|.94751；8/11|
|[3](evidence/region-medoid/sample3.json)|20/22/24；19.36%|20/22/24；21.47%|20/22/24；18.25%|.97318；7/11|
|[5](evidence/region-medoid/sample5.json)|9/11/12；17.01%|9/10/12；18.46%|9/11/12；17.11%|.98031；8/11|

2 号相对原算法多找到 #87，双方原来就匹配的 42 个孔 P95 从 14.89% 升到 15.92%。3 号相对多数法多 #196、少 #139，共同孔 P95 从 21.86% 降到 17.62%；但对原 census 的相同 20 个孔仅从 19.36% 降到 18.25%。5 号三种方法只有同一批 9 个匹配。没有一件通过两种对照的全部筛选。CUDA 与独立 CPU 对每区域全部 128 个编码格、选择及两张输出的每一体素一致。2/3/5 有 152/362/148 个区域，多数法与区域法在 1,066/11,857/5,523 个体素上的预测不同。

### 真正的整区选择上限：只优化两项，而非十一项

[宽松目录](RESULTS-OBJECT-CATALOG.md)允许跨 12 项实验逐孔任选、忽略冲突和丢弃不利匹配。这里仅用实际选择器预设的 7 个输入，每区必须复制一个完整选项，不能丢弃匹配。核对全部边后，三件中均无能跨两个区域匹配的参考孔，因此可以精确枚举每个可达匹配数的最优 P95。动态规划只删除同一匹配数下尾部误差逐项不优的状态；生成的完整掩膜再次通过原生 8/37 分块、SciPy 标签与原一对一匹配，没有把理论上的孔拼成虚构的结果。

|开发试件|覆盖至少需要|最多能匹配|示例完整选择：匹配数，P95|能同时过参考覆盖和 P95 吗？|该选择通过全部十一项中的几项|
|---|---:|---:|---|---|---:|
|[2](evidence/region-headroom/sample2.json)|45/49|48/49|45，8.44%|能|10/11|
|[3](evidence/region-headroom/sample3.json)|22/24|23/24|22，14.48%|能|9/11|
|[5](evidence/region-headroom/sample5.json)|11/12|10/12|10，11.22%|不能；满足覆盖的 P95 不存在|7/11|

前两行是达到覆盖时 P95 的最低值。5 号的 11.22% **只适用于 10 个匹配**，不是达标成绩。#44 和 #62 在七组里都没有匹配；较宽的十二项目录曾匹配 #44。各匹配数精确分数、原始边、区域所有选项和逐孔 CSV 均公开；不发布知道答案才得到的掩膜当候选。重放从原档重新提取 3 个区域，两个目录除运行秒数外的 JSON 全同、6 份 CSV 字节相同。

[真实选择登记](REGION-MEDOID-PILOT.md) · [评分前封存的预测和区域选择](https://github.com/EtherealBian/cone-fdk/tree/f12dcfcf0b6f465b95a3b05ff671bb0fbbae8aa8/pores/evidence/region-medoid) · [上限登记](REGION-HEADROOM-AUDIT.md) · [真实预测逐孔表](evidence/region-medoid) · [理想上限逐孔表](evidence/region-headroom) · [三个始终未匹配的孔，在原图上是什么情况？](RESULTS-PERSISTENT-MISS.md) · [正式标准](FINAL-GATE.md)。

```powershell
python scripts/ptx.py pores/region_medoid.cu build/pores/region-medoid.ptx
python scripts/pore_region.py --out build/region-pred-new
# 评分读取分支中已经公开的预测，不读取诊断挑选的预测。
python scripts/pore_region_score.py --cache $cache --out build/region-score-new
python scripts/pore_region_headroom.py --cache $cache --out build/region-headroom-new
```
