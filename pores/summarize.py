"""Render the public acceptance table from the committed, unrounded per-scan reports."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def render():
    text = [
        "# 钴铬试件：逐件验收记录",
        "",
        "当前状态：**自动孔隙测量未获最终验收，不能作为工业尺寸认证。** 下列各行比较的是同一内部区域中的预测与 NIST 发布的数字分割，不是卡尺、标准球或破坏性测量真值。每个试件使用自身已标注的记录内校准切片；不宣称零标注迁移。",
        "",
        "正式停止条件见 [FINAL-GATE.md](FINAL-GATE.md)。基线先注册了 3、5 号试件；在修改方法后，2、3、5 均按开发试件处理。4、6 号为修改方法后的独立确认试件，不用于挑参数。没有通过开发门槛的候选不进入确认。",
        "",
        "| 方法 | 试件 | Dice | 孔隙率偏差（百分点） | 匹配 / 符合条件的预测 / 参考 | 逐孔体积误差中位数 / P95 | 结果 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for directory, name in [
        ("baseline", "阈值基线"),
        ("candidate-a", "A · 局部滤波"),
        ("candidate-b", "B · 重叠图块"),
        ("candidate-c", "C · 多尺度边界"),
    ]:
        for i in (2, 3, 5):
            fn = f"evidence/{directory}/sample{i}.json"
            report = json.loads((HERE / fn).read_text(encoding="utf8"))
            m, o = report["metrics"], report["objects"]
            passed = "达标" if report["final_passed"] else "未达标"
            text.append(
                f"| {name} | [{i}]({fn}) | {m['dice']:.5f} | {m['porosity_error_pp']:+.5f} | {o['matches']} / {o['eligible_prediction']} / {o['eligible_reference']} | {o['median_abs_volume_error_pct']:.2f}% / {o['p95_abs_volume_error_pct']:.2f}% | {passed} |"
            )
    text += [
        "",
        "正号表示高估，负号表示低估。匹配按 IoU ≥ 0.5 一对一计算；只纳入不接触区域边界且至少 64 个体素的连通体。未匹配的预测和参考都会降低覆盖率。表格仅作展示，门槛使用未四舍五入值；孔隙率接近不能抵消漏孔、误检或少数孔的较大体积误差。",
        "",
        "## 怎样复核一条结果",
        "",
        "1. 点击试件编号读 JSON（包括原始档案 SHA-256、区域坐标、所有门槛和阈值扰动结果）；同目录的 `*-pores.csv`、`*-matches.csv`、`*-cases.csv`（候选方法）分别是完整预测、符合条件的匹配和逐对象定位表。",
        "2. 在[实验附件](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-1)选择方法与试件，解压。打开 `*-cases.html` 查看漏检和体积误差最大的孔：左为原始局部切片，右为差异，旁边给出精确 TIFF 与 XYZ。`*-review.html` 可浏览区域内全部 128 层以及所有预测 ID。无需显卡或联网。",
        "3. 从独立复核器重算，而非相信 JSON 中的 pass 字段：`python pores/audit.py build/candidate-C --development 2 3 5 --cache <archive-directory>`。最终确认需另提供 `--development-dir` 和提前公开的冻结清单，强制核对 4 和 6；见[操作步骤](README.md#protect-the-confirmation-specimens)。",
        "",
        "## 明确区分两个误差来源",
        "",
        "直接输入发布掩膜，CUDA 与独立 SciPy 对试件 2、3、5 的 [187、303、112 个连通体](evidence/mask-oracle)逐行相符；同掩膜的孔隙体素数差为零，两种显存分块的对象表字节相同。这个结果只验证计量层，不能充当从原始灰度图像开始的检测精度。",
        "",
        "Windows 原生预览包只包含掩膜计量命令行及已知体素数的三对象样例（无 Python 运行时）；稳定发布仍为核桃重建器 [v0.1.0](https://github.com/EtherealBian/cone-fdk/releases/tag/v0.1.0)。",
        "",
    ]
    return "\n".join(text)


if __name__ == "__main__":
    (HERE / "RESULTS.md").write_text(render(), encoding="utf8")
