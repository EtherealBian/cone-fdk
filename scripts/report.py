"""Render the measured acceptance dossier from the JSON + JUnit outputs (no fitted values)."""

import hashlib
import json
import platform
import statistics
import subprocess
import xml.etree.ElementTree as ET

from common import REPORT, ROOT
from fetch import MD5


def load(name):
    return json.loads((REPORT / (name + ".json")).read_text())


def require(condition, text):
    if not condition:
        raise RuntimeError("Cannot publish acceptance report: " + text)


def pct(x):
    return f"{x * 100:.5f}%"


def mib(x):
    return f"{x / 2**20:.2f}"


def med(seq, k):
    return statistics.median(x[k] for x in seq)


def render():
    phantom, small, full, over, cold, profile = map(
        load,
        [
            "phantom",
            "walnut",
            "walnut-full",
            "oversize",
            "cold-benchmark",
            "batch-profile",
        ],
    )
    tests = ET.parse(REPORT / "native-tests.xml").getroot().find("testsuite").attrib
    require(
        int(tests["tests"]) >= 20
        and all(int(tests[k]) == 0 for k in ["errors", "failures", "skipped"]),
        "native test results",
    )
    for n, r in phantom.items():
        met = r["metrology"]
        limited = r["limited_run"]
        h = r["sha256"]
        require(
            len(r["native_runs"]) == len(r["astra_runs"]) == 3
            and r["astra_difference"]["rel_l2"] < 0.01
            and r["repeat"]["max_abs"] == 0
            and h["native"] == h["warmup"],
            "phantom precision or repeatability",
        )
        require(
            limited["peak_owned_bytes"] <= limited["budget_bytes"]
            and limited["slabs"] > 1
            and r["limited_difference"]["rel_l2"] <= 1e-5
            and limited["redzones_verified"],
            "slab budget",
        )
        require(
            met["bulk_rel_error"] < 0.03
            and abs(met["outside_mean"]) < 0.0015
            and met["inclusion_mean"] > 0.063
            and all(
                h["center_error_mm"] < 0.4 and h["diam_error_mm"] < 0.6 for h in met["holes"]
            ),
            "analytic geometry/attenuation",
        )
    require(set(phantom) == {"256", "512"}, "missing volume size")
    require(
        full["views"] == 720
        and small["views"] == 90
        and full["n"] == 512
        and full["offline_subset_bitwise_equal"]
        and list(full["archives_md5"].values()) == MD5
        and len(load("walnut-full-frames")) == 720,
        "full scan provenance",
    )
    require(
        all(
            r["corrected_vs_astra"]["rel_l2"] < 0.02
            and r["uncorrected_vs_corrected"]["rel_l2"] > 0.001
            for r in [full, small]
        ),
        "measured data disagreement",
    )
    o = over["native"]
    require(
        over["output_bytes"] == 4 * 1296**3
        and over["output_bytes"] > over["gpu_total_bytes_smi"]
        and o["peak_owned_bytes"] <= o["budget_bytes"]
        and o["slabs"] > 1
        and o["redzones_verified"]
        and len(over["checked_planes"]) == 5
        and all(p["relative_l2"] < 0.01 for p in over["checked_planes"]),
        "oversize proof",
    )
    require(
        all(
            len(v) == 3 and all(x["wddm_local_peak_bytes"] > 0 for x in v)
            for each in cold.values()
            for v in each.values()
        ),
        "WDDM benchmark",
    )
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    text = [
        f"""# ConeFDK 本机验收记录 · {tests["timestamp"][:10]}

**结论：以下预设门槛在本机实跑通过。** 独立 GPU 对照是 ASTRA {full["astra"]["astra_version"]} `FDK_CUDA` 的 Ram-Lak。精度使用全部体素、物理坐标与绝对衰减值，不乘拟合系数、不配准、不裁剪。真实样本没有实物尺寸真值。

- 平台：{platform.system()} {platform.release()}，build {platform.version()}；{gpu}。Python {platform.python_version()}；编译/数值依赖精确版本见 `requirements-win.lock`。
- 构建产物 SHA-256：EXE `{hashlib.sha256((ROOT / "build/cone-fdk.exe").read_bytes()).hexdigest()}`；PTX `{hashlib.sha256((ROOT / "build/kernels.ptx").read_bytes()).hexdigest()}`。
- 原生报告中的 `driver_api={o["driver_api"]}` 是驱动所支持的 API 版本，**不等于**生成 PTX 的 NVRTC 版本；本发布构建用 NVRTC 12.9 `compute_80`。cuFFT runtime 返回 `{o["cufft_version"]}`。

## 1 · 解析工件：实现间一致性及物理真值

180 个独立解析投影，160 列 × 128 行，源到等中心 120 mm、到探测器 205 mm、FOV 32 mm。两偏心通孔直径分别为 4.8 和 3.0 mm，夹杂远离中平面。阈值：ASTRA 全体积相对 L2 <1%；均匀材料误差 <3%；中心 <0.4 mm；等效直径误差 <0.6 mm。孔测量来自中心层的半衰减阈值，不代表设备测量不确定度。

|体积|对 ASTRA 相对 L2|材料衰减误差|孔中心误差 mm（两孔）|孔径误差 mm（两孔）|有限预算 / 块数 / 与整块最大差|
|---|---:|---:|---:|---:|---|"""
    ]
    for n, r in phantom.items():
        m = r["metrology"]
        s = r["limited_run"]
        holes = m["holes"]

        def fmt(k, holes=holes):
            return " / ".join(f"{h[k]:.5f}" for h in holes)

        text.append(
            f"|{n}³|{pct(r['astra_difference']['rel_l2'])}|{pct(m['bulk_rel_error'])}|{fmt('center_error_mm')}|{fmt('diam_error_mm')}|{s['budget_bytes'] // 2**20} MiB / {s['slabs']} / {r['limited_difference']['max_abs']:g}|"
        )
    text.append(f"""\n两档 native 热身结果与重复结果 SHA-256 一致；离面球中心区域衰减分别为 {phantom["256"]["metrology"]["inclusion_mean"]:.5f}、{phantom["512"]["metrology"]["inclusion_mean"]:.5f} mm⁻¹，真值为 0.075。输入、尾角批、实际分块参数和原始浮点指标见 [phantom.json](phantom.json)。

## 2 · 真实测量：由原始 TIFF 到完整 512³ 体

原始源：Alexander Meaney / University of Helsinki，Zenodo **6986012**，CC BY 4.0。8 个原始 ZIP 的 MD5 与发布者一致，使用 0001–0720，排除 0721 的 360° 重复终点。保留 [720 张原始 TIFF 的 SHA-256](walnut-full-frames.json)。探测器按强度作 8×8 面积平均；不是原始分辨率重建。90 角离线件与 720 角处理结果中每第八张逐元素相同（未校正、校正两套都核实）。FOV 40 mm；以原始探测器像素为单位循环左移 5 列，空气归一化取负对数。

|输入|输出|对 ASTRA 全体素相对 L2（门槛 <2%）|未校正与校正之间相对 L2|校正重建原生 CLI 内部壁钟|
|---|---|---:|---:|---:|""")
    for r in [small, full]:
        text.append(
            f"|{r['views']} 角，280×296|{r['n']}³|{pct(r['corrected_vs_astra']['rel_l2'])}|{pct(r['uncorrected_vs_corrected']['rel_l2'])}|{r['native'][1]['wall_ms'] / 1000:.3f} s|"
        )
    text.append(f"""\n[实测正交切面](walnut-full-orthogonal.png) · [未校正 / 校正 / ASTRA](walnut-full.png) · [离线阅片](https://github.com/EtherealBian/cone-fdk/releases/download/v0.1.0/walnut-512-viewer.html) · [原始日志](walnut-full.json)。校正前后的差异说明操作生效，不是空间分辨率或实物精度的标定。校正体 SHA-256：`{full["output_sha256"]}`。

## 3 · 相同机器、相同输入的 GPU 性能对照

阶段时间和进程端到端**分别**统计。各分辨率先对两实现热身，再各运行 3 次，中位数如下。GPU Event 只覆盖 native 相应阶段；ASTRA 算法调用包含其内部传输/资源管理，不能从这张表推出纯内核加速比。`native wall` 含 CUDA 初始化、主机输入校验和输出 flush；ASTRA I/O+算法自读投影前起计到写盘和清理结束，不保证有同样的 flush。

|体积|native 滤波 / 反投影 ms|native H2D / D2H ms|native wall ms|ASTRA 算法调用 ms|ASTRA I/O+算法 ms|
|---|---:|---:|---:|---:|---:|""")
    for n, r in phantom.items():
        a = r["native_median"]
        b = r["astra_median"]
        text.append(
            f"|{n}³|{a['filter_ms']:.2f} / {a['bp_ms']:.2f}|{a['h2d_ms']:.2f} / {a['d2h_ms']:.2f}|{a['wall_ms']:.1f}|{b['algorithm_seconds'] * 1000:.1f}|{b['io_algorithm_seconds'] * 1000:.1f}|"
        )
    text.append("""\n另开进程，双方采用相同外层计时器，每档交替顺序，各 3 次；是**新进程**，不清 OS 文件缓存/驱动 JIT 缓存。ASTRA 的 Python 启动和导入计入该列。通过 Windows WDDM 以约 10 ms 间隔采样实际工作进程的 local GPU memory，表中列出各次峰值的中位数，可能漏掉瞬时尖峰。native 精确计数只覆盖引擎持有的设备缓冲（含 cuFFT 工作区和边界），不包含驱动/上下文。

|体积|新进程端到端 native / ASTRA s|WDDM 峰值 native / ASTRA MiB|native 自有峰值 MiB|
|---|---:|---:|---:|""")
    for n, r in cold.items():
        a, b = r["native"], r["astra"]
        text.append(
            f"|{n}³|{med(a, 'elapsed_seconds'):.3f} / {med(b, 'elapsed_seconds'):.3f}|{mib(med(a, 'wddm_local_peak_bytes'))} / {mib(med(b, 'wddm_local_peak_bytes'))}|{mib(a[0]['metrics']['peak_owned_bytes'])}|"
        )
    a, b = profile
    text.append(f"""\n256³ 的单次 batch=1→16 消融：反投影启动 {a["launches_bp"]}→{b["launches_bp"]} 次，反投影 {a["bp_ms"]:.2f}→{b["bp_ms"]:.2f} ms；这是一次观测，不是跨机保证。[冷进程明细](cold-benchmark.json) · [批量消融](batch-profile.json)。

## 4 · 真正大于整卡的输出

`1296³ × float32 = {over["output_bytes"]:,} B`，显卡物理显存 `{over["gpu_total_bytes_smi"]:,} B`。512 MiB 自有预算、实测峰值 **{mib(o["peak_owned_bytes"])} MiB**，每块最多 {o["slab"]} 层，共 **{o["slabs"]} 块**。native 内部壁钟 {o["wall_ms"] / 1000:.3f} s。确实落盘写出整个体，再读回并哈希；不是估算。SHA-256 `{over["sha256"]}`。

独立 ASTRA 校验的层号为 {", ".join(str(p["z"]) for p in over["checked_planes"])}，包含相邻接缝；最差切面相对 L2 **{pct(max(p["relative_l2"] for p in over["checked_planes"]))}**。这五层不是全体积比较。[原始记录](oversize.json)。

## 5 · 回归与未验证项

- [JUnit](native-tests.xml)：**{tests["tests"]} passed，0 skipped**。覆盖异常退出不替换原输出、短/长文件、非有限值和有限数溢出、非法几何、路径和硬链接别名、中文路径、角度反向、非等距探测器像素、自动分块与尾批。
- 小体积测试逐体素对独立 FP64 空间域实现：直接卷积（不使用 FFT）、射线与探测器平面相交、双线性零延拓；覆盖探测器边缘脉冲、非对称随机数据与最小 2×2 探测器，门槛为相对 L2 <5×10⁻⁶、峰值归一最大误差 <10⁻⁵。该实现只用于测试。
- 离线 HTML 另有回归：校验预览层号、PNG 的尺寸、解码与完整性，且不允许模板替换污染二进制载荷。发布包在干净目录的演示复测记录单列于包旁的 `verification.json`。
- 本次成功输出均验证 GPU 缓冲前后边界哨兵；它不能覆盖所有越界读写。另尝试的 NVIDIA Compute Sanitizer 未能附着目标进程；[原始日志](sanitizer-memcheck.txt)。**不将 memcheck 计为通过，也不声称已做完整内存安全证明。**
- 在这台 Windows 单 GPU 上复现，不构成跨设备认证、缺陷检出概率或计量/临床合规证明。""")
    file = REPORT / "RESULTS.md"
    file.write_text("\n".join(text) + "\n", encoding="utf8")
    print(file, flush=True)


if __name__ == "__main__":
    render()
