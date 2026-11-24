# 同一原始层上，论文的黑白插图能当作分割标签吗？

**在登记的两个区域都不能。** 试件 2 第 447 层的原图和中值图能用同一几何定位，但这不足以说明论文最后一幅黑白插图与发布的三维分割逐体素一致。这里没调阈值、没重新对齐插图，也没在两幅图中挑更有利的一幅当标签；测量参考仍是 NIST 分割档。

| 区域 | 原生中心数 | 归档为参考、插图为预测：TP / FP / FN / TN | Dice | 精确率 / 召回率 | 归档与固定算法 Dice |
|---|---:|---|---:|---|---:|
| 左（先前灰度配准的拟合侧） | 7,038 | 102 / 36 / 43 / 6,857 | .72085 | .73913 / .70345 | .90034 |
| 右（未用于拟合） | 13,464 | 923 / 124 / 105 / 12,312 | .88964 | .88157 / .89786 | .97889 |

两侧都有足够的前景和背景，却都没有达到事先登记的 Dice ≥.97、精确率和召回率各 ≥.95。右侧归档与算法原有 44 处分歧，插图在 23 处与归档相同、21 处与算法相同；另有 208 处归档与算法一致而插图不同。左侧分别为 16、13 和 66 处。右侧归档与插图的 229 处分歧中，有 132 处灰度在 16–239，余下 97 处不在该区间。左侧是 39/79。所有色调都纳入评分。**这些数字无法区分出版时缩放、处理步骤差异或其他成因，也不证明归档错误。**

[事前协议](https://github.com/EtherealBian/cone-fdk/blob/483fa59/pores/FIGURE-BINARY-PROVENANCE.md) · [查阅末图前公开的坐标、归档与固定预测](https://github.com/EtherealBian/cone-fdk/tree/750d180/pores/evidence/figure-binary) · [完整八格、三个两两比较](evidence/figure-binary/check.json) · [独立原档重放](evidence/figure-binary/replay.json)。

独立路径从固定 PDF 页 6 的原始图像对象直接读灰度，从官方 raw/seg ZIP 读 3 个灰度平面和 1 个标签平面；沿显示坐标枚举并反查原生中心，而首条路径遍历原生网格。两者在 **20,502 个唯一中心**上的坐标哈希、归档判定和八格计数全部相同。独立 CPU 路径另以有效邻域的 27 点中值、双精度窗口积分和指数权重、标量圆盘 Bernsen 从原始强度重算；两区固定算法的最终判定均为 **0 处差异**。该核对限于一个开发截面，不能替代三维逐孔或保留试件验收。

```powershell
# 从仓库根目录运行；$pdf 为论文作者版（哈希在脚本），PDF 在本地，不分发面板像素。
python scripts/pore_last_panel.py $pdf build/panel-new.png
python scripts/pore_figure_binary.py score --panel build/panel-new.png --out build/panel-score-new.json
# 第二条路径只需 NumPy、Pillow、pypdf；不读上一步生成的 PNG，不使用 CUDA。
python scripts/pore_printed_target_audit.py --pdf $pdf --cache $cache --out build/panel-replay-new.json
```
