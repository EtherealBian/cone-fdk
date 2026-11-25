# 只带走争议位置：离线选区复核

同事问「跨接缝的这 36 格，究竟是不是同一片？」不必把整张标签图和 GPU 环境都交给他。发给他这块选区的证据、两条可信摘要和选区坐标；他用标准 Python 即可读出 **36 格全属于原图区域 2521**。编号仍指向整张图的最早超限格，裁剪不会重新编号。

| 可下载的模拟作业示例 | 坐标 `轴起行 方位起列 高 宽`（从 0 起） | 选区结果 |
|---|---|---|
| [跨 359/0 列的接缝](roi-seam-receipt.zip) | `18 354 3 12` | 36 格实测超限，同一 ID；没有盲格 |
| [不可测对照的一小段](roi-blind-receipt.zip) | `8 12 5 21` | 105 格中 14 格实测低于门槛、91 格不可测。已观察到的超限格为 0；不能据此断言此处无缺陷 |

每块示例只重读了两类载荷合计 **14,400 B**（8 个整行），而原图两类载荷为 **73,800 B**。不含索引、清单或 JSON 文本。**这不是 ZIP 压缩率承诺**：小图的整图压缩包实际上可能更小。适用场景是从较大、已有可信索引的数据中抽查和移交某些位置。证据中包含整个相交行带以及完整父清单，不是脱敏裁剪。

## 复核者：只需要标准 Python

从本仓库取 `scripts/periodic_roi.py` 和相应 ZIP，在新目录解压。命令中的摘要来自[资格记录](periodic-roi-qualification.json)，而不是取自待验 ZIP。`-I -S` 隔离用户路径并不加载 site-packages；复核不启动 CUDA、不加载 NumPy/SciPy、不访问投影或原标签图。

```powershell
Expand-Archive reports/roi-seam-receipt.zip build/seam-receipt
python -I -S scripts/periodic_roi.py verify build/seam-receipt --parcel-pin 961648e72ce25d1e3d36b1f8a576af3e63750a2b809ca4e160a183e075ba2395 --index-pin f01b3a044cf6a07877295d0bda6fccc567bb8ce1ce6e6c3f07bc10e7d894a7b6 --box 18 354 3 12

Expand-Archive reports/roi-blind-receipt.zip build/blind-receipt
python -I -S scripts/periodic_roi.py verify build/blind-receipt --parcel-pin 90cb28cac64a07fde8e5a64bc721adc5762e44fe3ab10b71bdc896967fa1e649 --index-pin abeff23c931180bbe91c9544d6b0a7299832736ef1ba5a20a9c2997e335a5392 --box 8 12 5 21
```

完整父包和索引各有一个外部可信摘要，接收方还应独立知道要复核的坐标。程序验两条摘要、索引与父清单的关联，以及每一个相交行带的掩码和标签；从同一份已核对字节重算分类、区域号计数及按选区顺序的两个摘要。方位列可跨缝最多一周，轴向绝不回绕。它没有读取未相交的行，不会把那些行或整个试件标成合格；要复核全图标签和拓扑，使用[完整标签包验证](PERIODIC-LABELS.md)。

## 发送者：认证整图一次，按需出证

在已解压的完整示例上执行（见上文链接的完整包说明）。`index` 调用带 SciPy 的全图证书校验，然后重新流式取块、同时计算完整载荷的两个摘要，防止两遍之间源文件被替换。建议先把索引摘要经可信渠道交给复核方，而不是只附在 ZIP 里。`pack` 在 Windows 以新目录发布，不覆盖已有证据。只需在本机查结果可用 `query`，无需生成新目录。

```powershell
$parcel = '961648e72ce25d1e3d36b1f8a576af3e63750a2b809ca4e160a183e075ba2395'
$index = python scripts/periodic_roi.py index build/example-seam build/seam.index.json --parcel-pin $parcel --block-rows 4
python -I -S scripts/periodic_roi.py pack build/example-seam build/seam.index.json build/my-receipt --parcel-pin $parcel --index-pin $index --box 18 354 3 12
```

## 验收及复现

[事先固定的规程](../fixtures/periodic-roi-receipt-protocol-v1.json)要求对 67 张已认证图分别做四块无重叠分割，加一个底部单格探针。**335 次查询**与 NumPy 直接索引两份原始数组的结果一致，分割恰好覆盖 **988,920 个配对格**。两条拒测来源没有索引。两个示例均从 ZIP 重新解压，用隔离标准库进程复算；[完整回归](periodic-roi-regression.json)另查来源、索引、协议与代码摘要。局部改动、错摘要、混换行块、错误位置、缺件、多件和发布冲突均有反例；另用读操作记录核实未扫全图。

要重做发布的确切字节，使用[原管线重放](PERIODIC-LABELS.md)保留下来的 `build/my-sources` 及相同的已资格认定的程序和驱动，生成时引用**归档的设备资格文件**。另跑性能实验所得新时间会改变资格文件摘要及其所有下游摘要，不能当成同一个父包。

```powershell
python scripts/qualify_periodic_parcels.py build/my-sources/fixtures/stripe-campaign-plan.json reports build/periodic reports/periodic-label-qualification.json build/archived-parent-parcels
python scripts/qualify_periodic_roi.py reports/periodic-parcels.json build/archived-parent-parcels build/my-roi-replay
```

这项验收比较的是离散图和证据字节。它不提供毫米精度、实物缺陷概率或整件合格判定。
