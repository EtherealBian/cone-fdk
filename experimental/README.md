# 重建样例与复核

## 运行随仓核桃样例

在仓库根目录运行，依赖 NumPy。随仓为 90 角经校正的 FP32 投影；它**不是** README 中的 720 角计时输入。输出轴序为 `[z,y,x]`，x 最快。

```sh
python3 -c "import numpy as np; np.load('experimental/fixtures/walnut-s8-b8/corrected.npz')['views'].tofile('build/walnut.f32')"
./build/cone-fdk experimental/fixtures/walnut-s8-b8/corrected.geom build/walnut.f32 128 40 build/walnut128.f32
ctest --test-dir build --output-on-failure
```

## 数值复核

在同一 GB10 上重新运行旧 CLI 与新 CLI，按字节比较原始小端 FP32 文件；不取切片、不容忍差、不做强度拟合。三组输入均为实测投影，角度批大小 16。

| 样本 | 角数 | 网格 / FOV | 相同输出的 SHA-256 |
|---|---:|---|---|
| 核桃 | 720 | 512³ / 40 mm | `868249894840a28eb9db08dd75b36c56e23aab83155fad38f8a971ed734be25f` |
| 松果 | 90 | 256³ / 44 mm | `616067b46deaaf91ae5de13d678e5eb31f6bfc94d5cdc77cb59d069691dc71a0` |
| 松果 | 720 | 256³ / 44 mm | `3eea97fc5c4861d6cdc15a12fe2399883456402f3b68c43e48bfd4533b3422ee` |

通过 `sha256sum` 复核以上完整输出。另附[核桃全体积 ASTRA 对照](reports/gb10/walnut-astra.json)与[三网格六组配对原始数据](reports/gb10/multigrid-timing.json)。
