"""Regenerate the C/D development comparison from published object matches and reports."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

PORES = Path(__file__).resolve().parents[1] / "pores"


def read(base, i):
    return json.loads((base / f"sample{i}.json").read_text(encoding="utf8"))


def pairs(base, i):
    with (base / f"sample{i}-matches.csv").open(newline="") as f:
        return {
            int(row["ref_scipy_label"]): float(row["signed_volume_error_pct"])
            for row in csv.DictReader(f)
        }


def render():
    root = PORES / "evidence"
    c = root / "candidate-c"
    d = root / "candidate-d"
    ab = root / "articulation-c"
    scans = (2, 3, 5)
    lines = [
        "# 三维孔颈实验：拆开了两个孔，逐件验收仍未通过",
        "",
        "候选 D 将相邻切片的证据加入 CUDA 滤波后的孔隙边界估计。[方法和案例](EXPERIMENT-D.md)在评分前登记。以下均为开发试件，输入是已重建的 NIST 钴铬 XCT；每件用自己的独立标注切片做校准。[原定 11 项门槛](FINAL-GATE.md)与孔匹配规则不变。",
        "",
        "| 试件 | 方法 | Dice | 孔隙率偏差（百分点） | 匹配 / 合格预测 / 合格参考 | 匹配孔体积绝对相对误差：中位 / P95 | 通过项 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for i in scans:
        for p, meth in ((c, "C"), (d, "D")):
            r = read(p, i)
            o = r["objects"]
            met = r["metrics"]
            g = r["final_gates"]
            lines.append(
                f"| [{i}](evidence/{p.name}/sample{i}.json) | {meth} | {met['dice']:.5f} | {met['porosity_error_pp']:+.5f} | {o['matches']} / {o['eligible_prediction']} / {o['eligible_reference']} | {o['median_abs_volume_error_pct']:.2f}% / {o['p95_abs_volume_error_pct']:.2f}% | {sum(g.values())}/{len(g)} |"
            )
    lines += [
        "",
        "P95 上限为 15%；三件都未达到。每件的孔隙率绝对和相对误差都达到目标。试件 5 的 D 虽有较高 Dice，15 个合格预测只匹配到 10 个，不能把整体体素重合当成逐孔可靠。未匹配对象不参与体积百分位，单列在覆盖率中。",
        "",
        "### 只看 C 和 D 都匹配到的同一批参考孔",
        "",
        "| 试件 | 共同匹配 | D 新匹配 / 丢失 | 共同孔 P95（C → D） | 绝对体积误差变小 / 不变 / 变大 |",
        "|---|---:|---:|---:|---:|",
    ]
    for i in scans:
        u, v = pairs(c, i), pairs(d, i)
        ids = sorted(u.keys() & v.keys())
        a, b = (np.array([abs(x[k]) for k in ids]) for x in (u, v))
        lines.append(
            f"| {i} | {len(ids)} | {len(v.keys() - u.keys())} / {len(u.keys() - v.keys())} | {np.percentile(a, 95):.2f}% → {np.percentile(b, 95):.2f}% | {sum(b < a)} / {sum(a == b)} / {sum(b > a)} |"
        )
    lines += [
        "",
        "这张配对表用于区分边界改善和匹配对象变化；不替代前表的全量验收。",
        "",
        "## 回到试件 3 的原始切片",
        "",
        "![试件 3 原始灰度、NIST 标注、C 的合并预测和 D 的两个独立预测](evidence/candidate-d/sample3-121-127.png)",
        "",
        "`set1sample3raw_0455.tif`，X=[378,407)，Y=[600,629)，零基坐标。四格取自相同的 29×29 像素区。蓝：参考 121 或对应的 D 孔；黄：127 或对应的 D 孔；粉：C 的同一连通体。体积数来自完整三维对象，而不是数图上的像素。原始数据：Kim 等，NIST，[10.18434/M32162](https://doi.org/10.18434/M32162)。",
        "",
        "- C 的预测 3405094 有 436 个体素，一对一规则只能匹配 121（参考 296 个体素，高估 47.30%）；对 127 没有独立匹配。",
        "- D 分别预测 278 和 108 个体素；对照 296 和 128 个体素，误差为 −6.08% 和 −15.63%（后者精确值 −15.625%）。另两个参考孔 177、199 仍与同一裁边预测相交，不能算独立匹配。详见 [C](evidence/candidate-c/sample3-overlap.csv) 与 [D](evidence/candidate-d/sample3-overlap.csv) 的完整体素交集表。",
        "",
        "## 对照：只切一颗体素能修复吗？",
        "",
        "[预先规定的单体素切断操作](ARTICULATION.md)只读取 C 预测掩膜，不访问参考来选择切点；对每个不碰裁边的原对象至多切一次，切后两侧都必须至少 64 个体素。它仅用来验证机制，不能冻结为正式预测器。",
        "",
        "| 试件 | 切点 | 发生变化的参考匹配 | 匹配孔体积误差 P95 |",
        "|---|---:|---:|---:|",
    ]
    for i in scans:
        s = read(ab, i)
        lines.append(
            f"| [{i}](evidence/articulation-c/sample{i}.json) | {s['cut_count']} | {len(s['changed_matches'])} | {s['objects']['p95_abs_volume_error_pct']:.2f}% |"
        )
    lines += [
        "",
        "试件 3 的并孔无法靠这一规则拆开；试件 2 唯一的切点反而使参考 44 失去匹配。不将这项后处理纳入发布管线。",
        "",
        "## 复核入口",
        "",
        "- [D 的三个离线 ZIP 与 SHA-256](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-2)：解压后先开 `*-cases.html` 按误差找孔，再开 `*-review.html` 核对整幅切片。原始灰度查看器是固定窗宽显示，掩膜无损。",
        "- [D 的 JSON / CSV](evidence/candidate-d)：`*-overlap.csv` 枚举任一端合格的所有非零交集，另记完全无交集的合格对象（对端 ID 0）；两列辅助计数为交集达到对端至少 1/4 且对端 ≥64 体素的对象数，含裁边对象。它不是正式的一对一匹配。`*-size-bands.csv` 按参考孔体素数分段列出误差。",
        "- 复现 D 的预测实现见 `65db5e5`。`python pores/audit.py build/candidate-D --development 2 3 5 --cache <archive-directory>` 从源档、复核掩膜与组件表重算；三件均重放一致，因数值不达标返回 1。`freeze.py` 拒绝生成确认清单。试件 4、6 未评分。",
        "- [基线与 A/B/C](RESULTS.md) · [稳定的核桃重建器 v0.1.0](https://github.com/EtherealBian/cone-fdk/releases/tag/v0.1.0)。这里检验与数字分割的一致性，不是物理尺寸认证。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    path = PORES / "RESULTS-D.md"
    text = render()
    if args.check:
        if path.read_text(encoding="utf8") != text:
            raise SystemExit("Stale D report")
    else:
        path.write_text(text, encoding="utf8")
