# 哪一份外部扫描值得索取？

**优先索取 Iuso 等人一件 CoCr-DG1 圆柱的配准灰度与数字分割，另选 316L 立方体的人工标注作为跨材料对照。** 两组均未收到，不参与训练或计分。[E 的逐件结果](RETAINED-BASELINE.md)和[最终验收协议](FINAL-GATE.md)不变。

|来源|原始记录能证实什么|本项目的处理|
|---|---|---|
|Iuso 等人的金属增材制造 XCT|[论文](https://doi.org/10.1007/s10489-024-05647-z)第 2、3.1 节及 Data Availability；[作者稿 v2](https://arxiv.org/pdf/2305.07894v2)。研究中有两件 CoCr-DG1 圆柱。圆柱标签来自三维图像处理；单独的 316L 立方体采用人工标注。论文注明可向通讯作者合理索取；[公开代码的固定版本](https://github.com/snipdome/nn_3D-anomaly-detection/tree/d126468d0827ec772e82e3bb3c7e0c75810e5423)没有扫描体文件。|CoCr 更接近原任务，但只能比较数字分割；立方体可检验边界估计对另一种标注的敏感性，不能充当 CoCr 实物真值。尚未核验任何体素、完整孔数或标注覆盖范围。|
|NIST 3D-GAN 未熔合孔（[数据 DOI](https://doi.org/10.18434/mds2-3963)；[政府目录来源说明](https://catalog.data.gov/dataset/a-dataset-of-synthetic-additive-manufacturing-lack-of-fusion-pores-generated-using-a-three)）|目录明确将 [10.18434/M32162](https://doi.org/10.18434/M32162) 列为提取训练孔的源扫描，正是本项目的 CoCr 数据集。3,503 是合成孔和训练孔合计，不是试件数；发布的三维孔为二值形状。|仅查目录，不下载孔形、权重或图像。不能当独立新试件；在无法排除 4/6 来源时，生成的孔也不拿来训练或选参数。不能由目录反推某个孔一定来自 4/6。|
|XCTPore（[作者说明](https://github.com/BismaMutiargo/XCTPore/blob/aa8e71b8ed1b2c63ba599da2d059b5323710d752/README.md)，[论文](https://doi.org/10.58286/27521)）|作者说明为每栈 300 张灰度及对应掩膜，官方桶下载费用由请求方承担。另有公开的[划分清单](https://huggingface.co/datasets/kahmed68/xctpore-canonical-splits/tree/bfd64e4195326f23ef3a96929e993c3de02d3d29)，其中没有灰度和标签体。|没有使用付费账户，没有下载原卷。数组第一维为 300 不能代替源 z 步长、物理间距及标签来源说明；在这些信息核实前，不从切片数推断三维孔体积。|

### 可直接使用的数据请求草稿（未发送）

> For a reproducible pore-volume comparison, would it be possible to share a contiguous interior subvolume from one of the CoCr-DG1 cylinders, together with its registered grayscale reconstruction and digital pore mask? A second subvolume from the manually labelled 316L cube would help us distinguish the two reference methods. Please include specimen ID, native voxel spacing and units, axis order, source slice indices, mask values, and any uncertain or unlabelled regions. A crop is sufficient if it contains complete pores and enough surrounding material to identify boundary-clipped ones. Could you also specify whether local research use, publication of derived measurements, and redistribution of any example images are permitted? We can publish aggregate results without redistributing the scans.

### 收到之后，先核查，再决定能不能用

1. 保存来源、授权、试件身份和文件摘要；若只允许本地处理，不向 GitHub 提交任何原图、掩膜或阅片包。核查与原 CoCr 集合的关系；来源不明的衍生数据隔离，不送进模型。
2. 核对两套文件的间距、单位、轴向、切片索引和原点；在相同坐标复核三个正交截面的配准。明确未标注区域，不把它当无孔材料。不能从匹配良好或同形数组反推配准正确。
3. 先在不看 E 预测的情况下固定开发区域、记录边界截断的孔、按原六连通口径清点完整且至少 64 体素的孔，再决定是否有足够样本。保留每个漏检、误检及已匹配孔的正负体积误差，不按误差方向挑区域。**不把外部试件的分数混入 2/3/5，不替代 4/6。**
