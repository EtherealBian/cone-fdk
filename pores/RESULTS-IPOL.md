# 上游去噪器复核：引用的原版算法能否缩小差距？

**没有。** 在同一组开发检查切片上，直接运行论文所引用的 IPOL 作者代码，试件 2、3、5 的 Dice 和召回率都低于原固定 CUDA 对照。不上正式三维测量，不把 IPOL 与历史 Avizo 视为同一个程序。

| 试件 | 校准选出的 σ（映射灰度） | Dice：固定 CUDA → IPOL | 精确率 / 召回率：IPOL | 共同假阳 / 共同漏检 |
|---|---:|---:|---|---:|
| 2 | 20 | .94405 → .93437 | .91378 / .95592 | 206 / 79 |
| 3 | 25 | .95829 → .94705 | .97063 / .92459 | 525 / 1,310 |
| 5 | 25 | .98033 → .97857 | .98326 / .97393 | 363 / 552 |

直接调用单线程 C++ 滤波函数，没有调用给原图重新加噪的演示程序。半径 2 的补丁、半径 10 的搜索、噪声方差校正、最大邻点权重作为自身权重、整块重叠平均均走原版实现。系数 0.6 是本实验对参数的解释，不能证明其与 Avizo 的 similarity 相同。常量平面逐点不变，另用独立双精度实现按输出位置反向收集 25 个补丁的投票，检查三个非均匀内部位置。两者在这三个位置均相差不到 0.001 原生灰度单位。输入没有先裁剪到 8 位。

[选择方法及区域](IPOL-FILTER-PILOT.md) · [开分前清单和 21 项校准分数](evidence/ipol-decisions.json) · [开分前 Git 提交](https://github.com/EtherealBian/cone-fdk/commit/af8d9c5) · [检查集全账](evidence/ipol-check.json) · [两种方法的封存掩膜](evidence/ipol-masks.zip)。

从[原始源码存档](https://www.ipol.im/pub/art/2011/bcm_nlm/revisions/2020-11-05/nlmeansC.tar.gz)下载到 `$archive`，在本提交的 Windows 环境用 MinGW-w64 g++（`$compiler` 为完整路径）和记录的 NIST 开发源档重放：

```powershell
python scripts/pore_ipol_filter.py predict --cache $cache --out build/ipol-replay --archive $archive --compiler $compiler
python scripts/pore_ipol_filter.py score --cache $cache --out build/ipol-replay
```

`predict` 对源码摘要进行校验，只从归档中读取六个白名单文件；它用已登记的 4 层校准标签选择 σ，并在不读此次 8 层检查标签的情况下生成掩膜。第一次 `score` 在预测提交已推送并经 GitHub API 确认后运行。复算应比较参数、掩膜及统计值；严格模式还核对工具和库摘要，跨编译器的新构建不冒充原构建。
