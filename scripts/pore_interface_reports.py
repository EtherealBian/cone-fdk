"""Render the two paired transfer reports from archived evidence, never from pooled specimens."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "pores"


def read(p):
    return json.loads(p.read_bytes())


def fmt(x, n=2):
    return "未定义" if x is None else f"{x:.{n}f}"


def count(a):
    o = a["objects"]
    return "/".join(str(o[k]) for k in ("matches", "eligible_prediction", "eligible_reference"))


def gate(a):
    return str(sum(a["gates_on_research_slab"].values()))


def ids(a):
    return "、".join(map(str, a)) or "无"


def render(stage, seal_id):
    e = ROOT / "evidence" / stage
    p = read(e / "predictions.json")
    s = read(e / "summary.json")
    rows = [read(e / f"sample{i}.json") for i in (2, 3, 5)]
    remote = "https://github.com/EtherealBian/cone-fdk"
    bits = "通过" if s["advance"] else "未通过"
    if stage == "orientation":
        text = [
            "# 换八个方向看同一块材料，哪些误判仍然存在？",
            "",
            f"**八方向平均{bits}三件共同筛选。** 用同一套已公开权重在横截面的八种翻转/转置下各预测一次，再把连续输出变回原坐标取平均。没有选最好的方向，也没有以投票多数替换平均输出。单方向对照逐体素还原了原 census 掩膜。",
            "",
            "|试件|匹配/合格预测/合格参考：单方向 → 八方向|共同孔数；体积绝对相对误差 P95|各自全部匹配孔 P95|Dice|通过项 / 11|",
            "|---|---|---|---|---|---|",
        ]
        for r in rows:
            i = r["specimen"]
            u, v = [r["arms"][k] for k in ("identity", "mean8")]
            c = r["screen"]
            text.append(
                f"|[{i}](evidence/orientation/sample{i}.json)|{count(u)} → {count(v)}|{len(c['shared_ids'])}；"
                + " → ".join(fmt(x) + "%" for x in c["shared_p95"])
                + "|"
                + " → ".join(fmt(a["objects"]["p95_abs_volume_error_pct"]) + "%" for a in (u, v))
                + "|"
                + " → ".join(fmt(a["metrics"]["dice"], 5) for a in (u, v))
                + f"|{gate(u)} → {gate(v)}|"
            )
        text += [
            "",
            "|试件|存在分歧的位置（1–7票）|八方向一致但漏检 / 误检|改对：找回孔 / 删假阳|改错：添假阳 / 漏检|只修分歧位置的理想 Dice 上限|新增 / 丢失匹配 ID|",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            i = r["specimen"]
            v = r["votes"]
            q = r["edits"]
            rec = p["specimens"][str(i)]
            c = r["screen"]
            text.append(
                f"|{i}|{sum(rec['vote_counts'][1:8]):,}|{v['unanimous_fn']:,} / {v['unanimous_fp']:,}|{q['fixed_fn']:,} / {q['fixed_fp']:,}|{q['new_fp']:,} / {q['new_fn']:,}|{fmt(v['dice_ceiling'], 5)}|{ids(c['gained'])} / {ids(c['lost'])}|"
            )
        text += [
            "",
            "理想上限只问：如果借助真值把所有有方向分歧的位置改对、保留所有一致判定，体素 Dice 最多是多少？它不是算法成绩，不约束逐孔体积 P95。每件 JSON 给出前景、材料各九个票箱和全部配对 ID。",
            "",
            "|试件|原生 CUDA vs 独立 FP64：视图 0…7 各自不一致数|平均后不一致数|逐视图最大绝对输出差 / 平均输出最大差|",
            "|---|---|---:|---|",
        ]
        for i in (2, 3, 5):
            r = p["specimens"][str(i)]
            text.append(
                f"|{i}|"
                + ", ".join(map(str, r["view_cpu_gpu_differences"]))
                + f"|{r['masks']['mean8']['cpu_gpu_differences']}|{max(r['view_max_error']):.3g} / {r['mean_max_error']:.3g}|"
            )
        text += [
            "",
            "两条卷积路径共用原始特征前端；原生连通体另按 8 / 37 层分块，两次 CSV 字节一致，并与 SciPy 逐行核对。"
            + ("" if s["advance"] else "按约定不再在这些分数上挑方向或改阈值。"),
            "",
            f"[事前协议](ORIENTATION-PILOT.md) · [查阅评分标签前公开的六张预测、三份票图和权重身份]({remote}/tree/{seal_id}/pores/evidence/orientation) · [逐件 CSV、评分和完整原始证据](evidence/orientation)。",
            "",
            "```powershell",
            "python scripts/pore_orientation.py --cache $cache --out build/new-orientation-pred",
            "python scripts/pore_orientation_score.py --cache $cache --out build/new-orientation-score",
            "```",
        ]
    else:
        text = [
            "# 给同一批边界位置标出距离，比只标孔和材料更准吗？",
            "",
            f"**距离监督{bits}跨试件配对筛选。** 只在原校准区内训练：同一网络、同一初始化、同一批次和增强，一臂只给出界面哪一侧（±0.5），另一臂给出到最近相反标签中心的距离减半体素，再加符号。仅使用距裁剪面至少四格、四格内见过相反类别的中心。它是数字标签的距离代理，不是实物表面位置。",
            "",
            "|试件|可监督中心：孔 / 材料|其中距相反中心 >1：孔 / 材料|含这两类较深中心的 z 层数（孔 / 材料）|",
            "|---|---:|---:|---:|",
        ]
        cal = read(e / "census.json")
        for i in (2, 3, 5):
            side = cal["specimens"][str(i)]["sides"]
            text.append(
                f"|{i}|"
                + "|".join(
                    " / ".join(f"{side[k][j]:,}" for k in ("pore", "material"))
                    for j in ("centers", "beyond_one", "planes_with_beyond_one")
                )
                + "|"
            )
        text += [
            "",
            "|试件|算法|匹配 / 合格预测 / 合格参考|全部匹配孔 P95|Dice|迁移区通过项 / 11|",
            "|---|---|---|---:|---:|---:|",
        ]
        for r in rows:
            i = r["specimen"]
            for k, name in (("census", "原基线"), ("sign", "仅符号"), ("distance", "有符号距离")):
                a = r["arms"][k]
                text.append(
                    f"|[{i}](evidence/distance-target/sample{i}.json)|{name}|{count(a)}|{fmt(a['objects']['p95_abs_volume_error_pct'])}%|{fmt(a['metrics']['dice'], 5)}|{gate(a)}|"
                )
        text += [
            "",
            "|试件|对照 → 距离|共同孔数|共同孔体积误差 P95|新增 / 丢失参考 ID|未满足的登记条件|",
            "|---|---|---:|---|---|---|",
        ]
        names = dict(
            support="共同孔数",
            p95_gain="P95 降至少 2 个百分点",
            dice="Dice",
            matched_fraction_reference="参考孔覆盖",
            matched_fraction_prediction="预测孔覆盖",
            all_p95="全匹配 P95 不升",
            replay="数值复核",
        )
        for r in rows:
            for k, c in r["comparisons"].items():
                text.append(
                    f"|{r['specimen']}|"
                    + ("原基线" if k == "census" else "仅符号")
                    + f" → 距离|{len(c['shared_ids'])}|"
                    + " → ".join(fmt(x) + "%" for x in c["shared_p95"])
                    + f"|{ids(c['gained'])} / {ids(c['lost'])}|"
                    + ("、".join(names[k] for k, v in c["criteria"].items() if not v) or "无")
                    + "|"
                )
        text += [
            "",
            f"两个训练方法各更新 1,536 次，各累计监督 {p['fits']['distance']['band_exposures']:,} 个带内中心（包含重复）；这不是独立样本量。共享 CUDA 原图特征后，原生卷积和 NumPy FP64 卷积逐体素比较，两种新方法各自的不一致数（仅符号 / 距离）："
            + "；".join(
                f"{i} 号 "
                + " / ".join(
                    str(p["specimens"][str(i)]["masks"][k]["cpu_gpu_differences"])
                    for k in ("sign", "distance")
                )
                for i in (2, 3, 5)
            )
            + "。最大绝对输出差、非零分歧的首个坐标见预测清单。另一个进程从相同数据重新训练，两套权重及训练清单逐字节一致；按球内 256 个非零偏移显式枚举，三件全部 4,718,592 个校准距离目标与距离变换的字节摘要也相同（[独立复核](evidence/distance-target/explicit.json)）。三种方法的原生对象表在 8 / 37 层分块间逐字节复现，并由 SciPy 核对。",
            "",
            f"[事先登记的方法与否决条件](DISTANCE-TARGET-PILOT.md) · [校准覆盖](evidence/distance-target/census.json) · [开分前公开的两套权重、六张预测及代码]({remote}/tree/{seal_id}/pores/evidence/distance-target) · [逐孔表与完整比较](evidence/distance-target)。",
            "",
            "```powershell",
            "python scripts/pore_distance.py fit --feature-dir build/axial-raw-dev --cache $cache --out build/new-distance",
            "python scripts/pore_distance.py predict --cache $cache --out build/new-distance",
            "python scripts/pore_distance_score.py --cache $cache --out build/new-distance-score",
            "```",
        ]
    text += [
        "",
        "以上只评已研究过的 z=[256,352) 迁移区。对照为 NIST 数字分割，不是对实物尺寸的认证。[正式验收](FINAL-GATE.md)分开报告，保留试件 4、6 未启用。",
    ]
    return "\n".join(text) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("orientation", "distance-target"))
    p.add_argument("seal")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if len(a.seal) != 40 or any(c not in "0123456789abcdef" for c in a.seal):
        p.error("Expected full immutable seal commit")
    doc = ROOT / ("RESULTS-" + a.stage.upper() + ".md")
    text = render(a.stage, a.seal)
    if a.check:
        if doc.read_text(encoding="utf8") != text:
            raise SystemExit("Stale report")
    else:
        doc.write_text(text, encoding="utf8", newline="\n")
