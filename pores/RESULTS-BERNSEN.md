# 相同灰度下，Bernsen 阈值的两个版本解释了多少差异？

**在三个开发试件的相同去噪字节上，ConeFDK 的严格规则与真实 Fiji 插件的 638,976 个判定逐个相同。** 改用论文印刷公式后三件 Dice 都没有提高。保留当前规则。此实验没有复现 Avizo 去噪器，也不能确认 NIST 当时使用的 Fiji 构建版本。

| 试件 | 与插件判定不同 | 换印刷公式改变的体素 | 新增假阳 / 找回漏检 | 两种规则都错：假阳 / 漏检 | Dice：插件 → 印刷公式 |
|---|---:|---:|---|---|---|
| 2 | 0 / 212,992 | 85 | 74 / 11 | 250 / 78 | .94405 → .93457 |
| 3 | 0 / 212,992 | 854 | 450 / 404 | 886 / 1,237 | .95829 → .95813 |
| 5 | 0 / 212,992 | 303 | 195 / 108 | 531 / 540 | .98033 → .97899 |

此次三件的改变全在高对比分支，全部从背景变为孔。不能把这批体素的结果外推到其他输入或阈值版本。特别是 3 号，印刷公式几乎抵消总体孔隙率偏差，却仍有超过两千个体素是两种规则都判错的。[预登记](BERNSEN-PROVENANCE.md) · [预测在开分前公开](https://github.com/EtherealBian/cone-fdk/commit/c5c34f0) · [输入字节、工具和掩膜摘要](evidence/bernsen-decisions.json) · [分支转移全账](evidence/bernsen-check.json) · [掩膜](evidence/bernsen-masks.zip)。

### 从上游代码复算

取 [Fiji 源文件（固定提交）](https://raw.githubusercontent.com/fiji/Auto_Local_Threshold/f5ad12b1edebf79c5ef656c0c7eeb83eb282952a/src/main/java/fiji/threshold/Auto_Local_Threshold.java) 和 [ImageJ 1.51h jar](https://repo.maven.apache.org/maven2/net/imagej/ij/1.51h/ij-1.51h.jar)。不分发这两份文件。脚本先核对 SHA-256，再编译未改动的原文件和单线程无界面适配器。在带 halo 的平面上执行；只比较登记的 8×256×104 个内部位置。固定 CUDA 对照在全部三件均还原先前的掩膜哈希。复算使用与清单相同的 Temurin JDK 21.0.1+12（`$jdk` 指向 bin 目录）、项目锁定环境和 NIST 开发源档。

```powershell
python scripts/pore_bernsen_provenance.py predict --cache $cache --out build/bernsen-replay --plugin $plugin --jar $jar --jdk $jdk
python scripts/pore_bernsen_provenance.py score --cache $cache --out build/bernsen-replay
```

印刷公式来源：[Kim 等，§4](https://doi.org/10.1016/j.addma.2017.06.011)。样本及数值换算参数：[NIST 数据集](https://doi.org/10.18434/M32162)。预处理仍然共用；任一方法都不是对实物尺寸的认证。
