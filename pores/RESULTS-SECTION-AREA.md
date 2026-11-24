# 截面面积能不能改善三维逐孔测量？

**三件均未通过预登记的开发筛选，不晋级正式候选。** 5 号的共同孔体积误差确实下降，但漏掉一个原先可匹配的孔；2、3 号的共同孔误差都上升。这不是 H 的正式区域，也不是对新试件的独立确认。

两臂共用相同初始参数、1,800 批基础训练数据、零阈值；唯一区别是面积臂每四步学习一次完整二维截面的面积。每臂只有一套跨三件共享的权重。训练用原 24 层；评估用另一个已研究过的 96 层区域 z=[256,352)。[预登记](SECTION-AREA-PILOT.md) · [开分前公开的权重和六个预测摘要](https://github.com/EtherealBian/cone-fdk/commit/0e73f09)。

| 试件 | 共同孔数 | 同一批孔体积误差 P95：对照 → 面积 | 匹配 / 预测 / 参考：对照 → 面积 | Dice：对照 → 面积 | 描述性过项数（11项） |
|---|---:|---:|---|---|---|
| [2](evidence/section-area/sample2.json) | 46 | 16.56% → 21.30% | 48/49/49 → 46/52/49 | .95134 → .93050 | 10 → 3 |
| [3](evidence/section-area/sample3.json) | 18 | 17.03% → 25.37% | 22/27/24 → 18/21/24 | .96752 → .96628 | 6 → 6 |
| [5](evidence/section-area/sample5.json) | 9 | 20.25% → 14.19% | 10/11/12 → 9/10/12 | .97696 → .97565 | 6 → 9 |

[5 号参考孔 **15** 的重叠明细](evidence/section-area/trace-5-15.json) 有 452 个体素。对照臂把它匹配到一个 416 体素的完整预测（交集 405，IoU 0.87473）。面积臂覆盖了其中 **415** 个体素，却把它们连在一个触及 ROI 边界的 **1,928** 体素对象上，IoU 只有 0.21120。多覆盖十个参考体素反而失去了独立对象。这是区域内匹配失败，不能据此声称真实孔消失。2 号丢失 ID 12、93；3 号丢失 71、125、153、186；三件均无新增匹配。逐孔表包括未匹配的参考 ID。

| 试件 | 可用三维父对象（触及 Z 面 / 任一面） | 可用二维截面 | 150 次抽取中不同父对象 / 不同截面 |
|---|---:|---:|---:|
| 2 | 43（21 / 22） | 269 | 41 / 94 |
| 3 | 46（34 / 35） | 909 | 44 / 79 |
| 5 | 18（16 / 16） | 307 | 18 / 74 |

截面不是独立三维样本。基础 patch 流和训练轨迹见[训练和预测清单](evidence/section-area/predictions.json)。独立再次训练得到相同的两份权重字节。每臂每件比较 6,291,456 个 CUDA 和 CPU 判定：六次均零差异；二者共享 CUDA 去噪输入。每件两臂的 8/37 层原生对象表逐字节相同，并由 SciPy 逐行复算；2、3、5 的行数分别为 125/153、189/228、91/121（对照/面积）。同掩膜的吻合不能替代与参考分割的精度验收。

### 重放

在本报告所在提交，按[环境说明](README.md)构建 CUDA；安装空间模型所用的 CPU PyTorch。下载[原始二值预测和两份权重](evidence/section-area/weights-masks.zip)，SHA-256 `f801d4c36c009240ed327123cfab76e089e1feed05ab63d2695a0ddf12f5f00c`。要从封存的掩膜和 NIST 源档重新测量，而非读取报告中的 pass 字段：

```powershell
Expand-Archive pores/evidence/section-area/weights-masks.zip build/section-replay
Copy-Item pores/evidence/section-area/predictions.json build/section-replay/predictions.json
python scripts/pore_section_score.py --cache $cache --pred build/section-replay --out build/section-rescored
python scripts/pore_section_trace.py --cache $cache --specimen 5 --ref 15 --out build/trace-5-15.json
```

要从头生成预测，在一个空目录依次运行 `scripts/pore_section_predict.py fit`、`infer --specimen 2`、`infer --specimen 3`、`infer --specimen 5` 和 `seal`；每条命令均带 `--cache $cache --out build/new-section`。比较权重和每个掩膜哈希，不把机器耗时当作数值精度。全部数据均为相对于 NIST 已发布数字分割的开发结果。4、6 不在本实验中。
