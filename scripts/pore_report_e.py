"""Rebuild the E development report with matched denominators and review-queue accounting."""

import argparse
import csv
import json
from pathlib import Path

from pore_pairs_de import load, paired, summarize

ROOT = Path(__file__).resolve().parents[1] / "pores"


def obj(p):
    return json.loads(p.read_text(encoding="utf8"))


def queue_counts(base, i):
    def read(suffix):
        with (base / f"sample{i}-{suffix}.csv").open(newline="", encoding="utf8") as f:
            return list(csv.DictReader(f))

    matches = {int(r["pred_id"]): r for r in read("matches")}
    counts = [0, 0, 0, 0]
    seen = set()
    for r in read("review-queue"):
        p = int(r["e_pred_id"])
        flag = r["flagged"] == "True"
        m = matches.get(p)
        if p in seen:
            raise ValueError("Duplicate review target")
        seen.add(p)
        bad = m is None or 100 * abs(int(m["pred_voxels"]) - int(m["ref_voxels"])) > 15 * int(
            m["ref_voxels"]
        )
        counts[(0 if flag else 2) + (0 if bad else 1)] += 1
    report = obj(base / f"sample{i}.json")["objects"]
    if len(seen) != report["eligible_prediction"] or matches.keys() - seen:
        raise ValueError("Incomplete review queue")
    return counts


def render():
    base = ROOT / "evidence/candidate-e"
    old = ROOT / "evidence/candidate-d"
    lines = [
        "# 候选 E：小孔边界的逐件复核",
        "",
        "[评分前登记的方法](EXPERIMENT-E.md)使用新的 CUDA 核函数，分别用最相似的约 8、32、96 个图像块估计灰度；校准时给小孔边界内外两侧加权。这里检验 NIST 钴铬 XCT 与**数字分割**的一致性，每件使用自己的少量标注校准。不是物理尺寸认证。",
        "",
        "## 原定 11 项门槛：逐件计算，不取三件平均",
        "",
        "| 试件 | 方法 | Dice | 孔隙率偏差（百分点） | 匹配 / 合格预测 / 合格参考 | 逐孔体积绝对误差中位 / P95 | 通过项 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    pairrows = []
    smallrows = []
    qrows = []
    fail = []
    for i in (2, 3, 5):
        for dir, tag in ((old, "D"), (base, "E")):
            r = obj(dir / f"sample{i}.json")
            m = r["metrics"]
            o = r["objects"]
            g = r["final_gates"]
            lines.append(
                f"| [{i}](evidence/{dir.name}/sample{i}.json) | {tag} | {m['dice']:.5f} | {m['porosity_error_pp']:+.5f} | {o['matches']} / {o['eligible_prediction']} / {o['eligible_reference']} | {o['median_abs_volume_error_pct']:.2f}% / {o['p95_abs_volume_error_pct']:.2f}% | {sum(g.values())}/{len(g)} |"
            )
        fail.append(
            f"- 试件 {i}：" + "、".join("`" + k + "`" for k, v in g.items() if not v) + "。"
        )
        a, b = summarize(paired(load(old, i), load(base, i)))
        pairrows.append([f"[{i}](evidence/candidate-e/sample{i}-paired-d-e.csv)", *a])
        smallrows.append([i, *b])
        qrows.append(
            [f"[{i}](evidence/candidate-e/sample{i}-review-queue.csv)", *queue_counts(base, i)]
        )

    def example(i, ref, status):
        with (base / f"sample{i}-paired-d-e.csv").open(encoding="utf8", newline="") as f:
            row = next(r for r in csv.DictReader(f) if int(r["reference_id"]) == ref)
        if row["status"] != status:
            raise ValueError(f"Review interpretation for sample {i} reference {ref}")
        return row

    lost = example(3, 197, "lost")
    gained = example(5, 67, "gained")

    def table(rows):
        return ["| " + " | ".join(map(str, row)) + " |" for row in rows]

    lines += [
        "",
        "逐孔误差以各参考孔自身的体积作分母。P95 上限为 15%。未匹配孔不参与百分位，但记入覆盖率；完整失败项如下：",
        "",
        *fail,
        "",
        "## 同一批孔有没有测得更准？",
        "",
        "| 试件 / 全量 CSV | 共同匹配数 | D 同孔 P95 | E 同孔 P95 | 误差变小 / 相同 / 变大 | 新匹配参考 ID | 丢失参考 ID | 两者均未匹配的参考 ID |",
        "|---|---:|---:|---:|---:|---|---|---|",
        *table(pairrows),
        "",
        "CSV 包含每个合格参考孔，列出预测体积减参考体积的体素差，以及仅预测、仅参考的体素数。未匹配的格子留空，不用零误差填补。",
        "",
        f"- **试件 3，参考孔 197：**参考 {lost['reference_voxels']} 体素，D 测得 {lost['d_pred_voxels']}；E 与它最重叠的连通体只有 {lost['e_miss_best_iou_pred_vox']} 体素。虽然 IoU 为 {float(lost['e_miss_best_iou']):.3f}，它仍低于 64 体素的资格线。这个孔记为漏检，不进入 E 的误差百分位。",
        f"- **试件 5，参考孔 67：**参考 {gained['reference_voxels']} 体素，D 的最佳重叠 IoU 为 {float(gained['d_miss_best_iou']):.3f}，未达到 0.5。E 新匹配到它，却测得 {gained['e_pred_voxels']} 体素（高估 {100 * int(gained['e_signed_delta_vox']) / int(gained['reference_voxels']):.2f}%）。所以共同孔的 P95 几乎没变，把新增孔纳入正式计分后，整件 P95 反而更高。",
        "",
        "### 放大看 64–127 体素的小孔",
        "",
        "| 试件 | 合格参考数 | D 匹配 | E 匹配 | 共同匹配 | D 共同孔 P95 | E 共同孔 P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *table(smallrows),
        "",
        "“—”表示没有共同样本；很小的样本数不支持稳定的尾部分布估计。完整分段表和物理单位换算见各件 `*-size-bands.csv`。",
        "",
        "## 哪些孔值得人工再看？",
        "",
        "[事先登记的复核规则](DISAGREEMENT.md)只比较 C、D、E 在同一位置的预测：找不到合格的同伴、IoU 不足 0.5，或任一同伴体积与 E 相差超过 E 的 15%，都标记出来。随后才用参考掩膜检查标记的效果。下表中“有问题”指 E 未匹配，或匹配但体积误差超过参考体积的 15%。",
        "",
        "| 试件 / 标记 CSV | 标记且有问题 | 标记但无此问题 | 未标记却有问题 | 未标记且无此问题 |",
        "|---|---:|---:|---:|---:|",
        *table(qrows),
        "",
        "未标记不能视为通过，多个方法可能犯同一个错误。E 完全漏掉的参考孔也不可能出现在按预测生成的队列中；完整验收始终包括这些漏检。",
        "",
        "## 复现与离线复核",
        "",
        "- [三个完整复核包与 SHA-256](https://github.com/EtherealBian/cone-fdk/releases/tag/measurement-evidence-3)。解压后在浏览器打开 `*-cases.html` 按误差找孔，或在 `*-review.html` 选择数字 ID 核对原始切片。`*-review-queue.html` 还可以在同一坐标并排查看原图及 C/D/E 三种预测。",
        "- 预测源码 `e50a3e8`；运行 `python pores/candidate.py --model E --specimens 2 3 5 --cache <archive-directory>`。包中的 CUDA 组件表与两种分块的字节一致；独立 `pores/audit.py` 可根据原始源档、复核掩膜和组件表重算门槛。",
        "- 从完整 C/D/E 复核目录运行 `scripts/pore_review_queue.py`；用 `scripts/pore_pairs_de.py` 从 D/E 的逐件匹配表和交集表重建配对 CSV。`scripts/pore_report_e.py --check` 检查本页与公开证据是否一致。",
        "- [校准网格对照](GRID-PILOT.md) · [切片方向对照](PLANE-PILOT.md) · [上轮 D](RESULTS-D.md) · [最终验收标准](FINAL-GATE.md)。开发未通过，预留的 4、6 未评分；稳定核桃重建版仍为 v0.1.0。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = ROOT / "RESULTS-E.md"
    s = render()
    if a.check:
        if path.read_text(encoding="utf8") != s:
            raise SystemExit("Stale E report")
    else:
        path.write_text(s, encoding="utf8")
