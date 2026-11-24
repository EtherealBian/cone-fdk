"""Build the research report or an all-three-fold, checksum-indexed offline archive."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEV = (2, 3, 5)
EVIDENCE = ROOT / "pores/evidence/transfer-pilot"
REPORT = ROOT / "pores/RESULTS-TRANSFER.md"
TAG = "boundary-transfer-pilot-1"
LINK = "https://github.com/EtherealBian/cone-fdk/releases/tag/" + TAG


def pct(q):
    return "—" if q is None else f"{q:.2f}%"


def ab(row):
    return " → ".join(map(pct, row))


def capacity(rec):
    """Post-hoc bound on direct size edits only, with the original matches held fixed."""
    rows = rec["base"]["references"]
    moves = {r["pred"]: r for r in rec["adjustments"]}
    results = []
    for rid, row in rows.items():
        v = row["voxels"]
        n = row["reference_voxels"]
        pid = row["pred"]
        if v is None:
            continue
        m = moves.get(pid)
        cap = round(0.25 * m["surface"]) if m else 0
        add = min(cap, m["outer"]) if m else 0
        residual = max(v - cap - n, n - v - add, 0)
        results.append((100 * residual / n, int(rid), n, v, v - cap, v + add))
    q = sorted(r[0] for r in results)
    if not q:
        return 0, 0, None, None
    j = (len(q) - 1) * 0.95
    low = int(j)
    high = min(low + 1, len(q) - 1)
    return (
        sum(100 * abs(n - v) > 15 * n for _, _, n, v, _, _ in results),
        sum(100 * max(left - n, n - right, 0) > 15 * n for _, _, n, v, left, right in results),
        q[low] + (j - low) * (q[high] - q[low]),
        max(results),
    )


def render(directory):
    data = [json.loads((directory / f"sample{i}.json").read_text(encoding="utf8")) for i in DEV]
    out = [
        "# 跨试件校准：别的试件能教会模型把这个孔量准吗？",
        "",
        "对每个开发试件取一段新的 96 层区域，用另外两个试件的完整孔学习边界该向内还是向外移动。候选 E 本身仍只用本件原来的 24 层训练。先公开[实验规则](TRANSFER-PILOT.md)和[三个概率体哈希](evidence/transfer-pilot/baseline.json)，再读取新区域标注（登记 `45ee74d`，预测存证 `7d11017`）。本页的 z=[256,352) 与原开发评分区域 z=[400,528) 不同；不能把两次结果当作同一批孔比较。",
        "",
        "| 目标试件 | 另一试件提供的拟合对数 | 共同小孔数 | 共同小孔体积误差 P95：修正前 → 后 | 全部匹配孔 P95：前 → 后 | 漏检：前 → 后 | 未匹配预测：前 → 后 | 筛选 |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    decisions = {"pass": "通过研究筛选", "fail": "未通过", "insufficient support": "样本不足"}
    for i, r in zip(DEV, data, strict=True):
        s = r["screen"]
        d = "，".join(f"{k} 号：{v}" for k, v in r["trained_on"].items())
        out.append(
            f"| [{i}](evidence/transfer-pilot/sample{i}.json) | {d} | {len(s['shared_ids'])} | {ab(s['shared_p95_pct'])} | {ab(s['full_p95_pct'])} | {s['missed'][0]} → {s['missed'][1]} | {s['unmatched'][0]} → {s['unmatched'][1]} | {decisions[s['decision']]} |"
        )
    ok = all(r["screen"]["decision"] == "pass" for r in data)
    out += [
        "",
        (
            "三件同时通过研究筛选；下一步仍须在原开发区域独立测量。"
            if ok
            else "**不采用这项修正。** 事先规定三个目标试件都要通过；不能只选改善的试件，也不能把每个试件的孔合并计算一个 P95。"
        ),
        "",
        "“共同小孔”是两个掩膜都匹配到的同一批 64–127 体素参考孔。每件至少需要 5 个共同小孔、每个供样试件至少 5 对；同孔 P95 须下降至少 3 个百分点，全部匹配孔 P95、漏检数、未匹配预测数均不能增加。正式 15% 门槛仍见[最终验收协议](FINAL-GATE.md)。",
        "",
        "## 新切片里有多少完整孔？",
        "",
        "| 试件 | 完整 32–63 / 64–127 / ≥128 体素 | 裁边 ≥64 | 可供其他件拟合的匹配对 | 基线 / 修正匹配数 | 共同小孔误差：变小 / 不变 / 变大 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in zip(DEV, data, strict=True):
        c = r["census"]
        t = [0, 0, 0]
        for k in r["screen"]["shared_ids"]:
            b, a = [r[m]["references"][str(k)] for m in ("base", "arm")]
            n = b["reference_voxels"]
            diff = abs(a["voxels"] - n) - abs(b["voxels"] - n)
            t[0 if diff < 0 else 2 if diff > 0 else 1] += 1
        out.append(
            f"| {i} | {c['complete_32_63']} / {c['complete_64_127']} / {c['complete_ge128']} | {c['clipped_ge64']} | {len(r['donor_pairs_for_other_folds'])} | {r['base']['objects']['matches']} / {r['arm']['objects']['matches']} | {' / '.join(map(str, t))} |"
        )
    out += [
        "",
        "每个供样对必须有完整预测和完整参考，预测为 32–256 体素、参考至少 32 体素，并符合一次性一对一 IoU 匹配。上表“可供拟合”与正式合格孔数不是同一个分母；没有把裁断的孔当完整体积。",
        "",
        "## 复核一个孔，或重放所有评分",
        "",
        f"1. 下载[三件合一的离线复核包]({LINK})，核对附件 SHA-256 后解压。打开 `sample2-slices.html`（或 3、5 号），选择参考 ID 或未匹配预测。滑块遍历全部 96 层；原始灰度、NIST 数字标注、E 基线和修正后掩膜显示在同一坐标。橙色为新增、紫色为删除体素。",
        "2. 压缩包里的 `*-review.npz` 保留未量化的 float32 概率，及无损的参考、基线、修正后三个掩膜。`*-pores.csv` 中的 `id` 是原生计量器的种子线性编号；JSON 中 `pred` 是 SciPy 扫描顺序标签，两者可用 CSV 的种子坐标关联，不能直接当作同一个编号。",
        "3. 从对应版本的仓库根目录，安装计量实验的锁定依赖。运行 `python scripts/pore_transfer_replay.py <解压目录> --viewers`：无需 GPU 即可用概率体和另两件的供样重新拟合，逐体素重放最终掩膜、检查组件表、匹配和评分。加 `--cache <NIST档案目录>` 可同时从原始 TIFF 核对所有参考体素和显示灰度。",
        "4. 从源档重做 GPU 实验：按测量分支说明编译 `pores/Build.ps1`；依次运行下列命令。所有目录参数使用实际路径，两个输出目录必须为空。",
        "",
        "```powershell",
        "python scripts/pore_transfer_predict.py --cache <cache> --out <baseline>",
        "python scripts/pore_transfer.py --cache <cache> --baseline <baseline> --out <out>",
        "Copy-Item <baseline>/baseline.json <out>/baseline.json",
        "python scripts/pore_transfer_view.py <out> --cache <cache>",
        "python scripts/pore_transfer_replay.py <out> --cache <cache> --viewers",
        "```",
        "",
        "CUDA 计算六面邻接壳层的唯一归属；勾连多个不同预测标签的背景体素不归任何一孔。用独立的 SciPy 腐蚀/膨胀逐体素核对内外边界，修正后重新做连通体分析。两个掩膜各用 8 层、37 层分块输出，同件同一掩膜的两张组件表必须字节一致。回归和评分在 CPU 上，未声称整套研究流程都是 CUDA。",
        "",
        "研究结果只描述与 NIST 数字分割的一致性。4、6 号确认区域未用于本实验；稳定版仍为核桃重建 v0.1.0。",
        "",
    ]
    audit = [
        "## 即使知道每个孔该增减多少，边界能挪得够远吗？",
        "",
        "事后诊断，不用于拟合或筛选：固定基线已匹配的孔和它们的 ID。给每个孔最有利的增减方向和数量，仍只允许实验规定的内外一层及 25% 边界上限；没资格调整的孔维持原体积。忽略连通体改变和新增/丢失匹配，所以这只是**直接修正体积的理想下界**，不是可部署模型的成绩，也不是其他分割方法的极限。",
        "",
        "| 试件 | 基线匹配孔中误差 >15% | 理想调整后仍必然 >15% | 理想 P95（固定匹配） | 剩余误差最大的参考孔：参考 / 基线 / 可直接调整区间（体素） |",
        "|---|---:|---:|---:|---|",
    ]
    for i, r in zip(DEV, data, strict=True):
        n, stuck, q, ex = capacity(r)
        witness = "—" if ex is None else f"#{ex[1]}：{ex[2]} / {ex[3]} / [{ex[4]}, {ex[5]}]"
        audit.append(f"| {i} | {n} | {stuck} | {pct(q)} | {witness} |")
    w = json.loads((directory / "sample5-merge.json").read_text(encoding="utf8"))
    main = w["reference"]
    found = {v["reference"]: v for v in w["overlaps"]}
    other = max((v["overlap_voxels"], k) for k, v in found.items() if k != main)[1]
    a, b = found[main], found[other]
    bg = w["reference_background_in_prediction"]
    miss = w["missed_reference_voxels"]
    pred = w["predicted_voxels"]
    n = w["reference_voxels"]
    if (
        sum(v["overlap_voxels"] for v in found.values()) + bg != pred
        or a["overlap_voxels"] + miss != n
        or data[-1]["base"]["references"][str(other)]["pred"] is not None
    ):
        raise ValueError("Merge witness no longer agrees with its masks")
    audit += [
        "",
        f"### 试件 5，第 {w['source_z']} 层：一个预测连到了两个参考孔",
        "",
        f"参考孔 #{main} 有 {n} 体素，E 的同一连通体却有 {pred} 体素。它覆盖本孔 {a['overlap_voxels']} 体素、另一个完整参考孔 #{other} 的 {b['overlap_voxels']} / {b['reference_voxels']} 体素，以及 {bg} 个参考背景体素，同时漏掉本孔 {miss} 体素。净高估 {b['overlap_voxels']} + {bg} − {miss} = **{pred - n} 体素（{100 * (pred - n) / n:.2f}%）**；#{other} 没有独立的一对一匹配。下一步应检验如何拆开被误连的孔，而不是继续给整个连通体统一增减体积。",
        "",
        "![同一张源切片：原始灰度、两个参考孔、单个预测和重叠归属](evidence/transfer-pilot/sample5-merge.png)",
        "",
        "[源坐标和交集计数](evidence/transfer-pilot/sample5-merge.json)。这张图选取一个二维四连通的预测区域确实同时覆盖两个参考 ID 的轴向切片；图上颜色只显示这一层，以上计数来自完整三维掩膜。",
    ]
    flags = json.loads((directory / "bridge-review.json").read_text(encoding="utf8"))
    audit += [
        "",
        "### 能否在没有参考标注时提醒人复核？",
        "",
        "[探索性提醒规则](BRIDGE-REVIEW.md)：在 E 的 0.5 连通体内查找彼此不连通的 0.90 高置信核心；至少两个核心各有 ≥32 体素且各占母体 ≥5% 才提醒。查看参考掩膜时，将同一合格预测覆盖两个不同完整参考孔各至少一半定义为对照事件。规则是在已看到 5 号案例后设计的，不能据此估计泛化性能。",
        "",
        "| 试件 | 合格预测连通体 | 提醒且符合事件 | 误报 | 漏掉的事件 | 未提醒且无此事件 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i in DEV:
        f = flags[str(i)]
        c = f["counts"]
        audit.append(f"| {i} | {f['eligible']} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |")
    audit += [
        "",
        "[全部预测孔、核心体积、覆盖参考 ID 与分母](evidence/transfer-pilot/bridge-review.json)。复核包下拉框会标出命中的母体；它只是提醒，**没有拆开误连，也没有改善任何计分**。对自己的概率图可运行 `python scripts/pore_bridge_review.py --probability <float32-ZYX.npy> --out queue.json`，不需要参考标签。",
    ]
    idx = out.index("## 复核一个孔，或重放所有评分")
    out[idx:idx] = audit + [""]
    return "\n".join(out)


def bundle(directory, out):
    names = ["baseline.json", "sample5-merge.json", "sample5-merge.png", "bridge-review.json"] + [
        f"sample{i}{suffix}"
        for i in DEV
        for suffix in (
            ".json",
            "-review.npz",
            "-slices.html",
            "-base-pores.csv",
            "-base-matches.csv",
            "-arm-pores.csv",
            "-arm-matches.csv",
        )
    ]
    extras = {
        "README.md": render(directory).replace("evidence/transfer-pilot/", "").encode(),
        "BRIDGE-REVIEW.md": (ROOT / "pores/BRIDGE-REVIEW.md").read_bytes(),
        "FINAL-GATE.md": (ROOT / "pores/FINAL-GATE.md").read_bytes(),
        "NOTICE.md": (ROOT / "NOTICE.md").read_bytes(),
        "TRANSFER-PILOT.md": (ROOT / "pores/TRANSFER-PILOT.md").read_bytes(),
    }
    sums = []
    with ZipFile(out, "x", compression=ZIP_DEFLATED, compresslevel=5) as z:
        for name in names + list(extras):
            payload = extras[name] if name in extras else (directory / name).read_bytes()
            sums.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
            z.writestr(name, payload)
        z.writestr("SHA256SUMS", "\n".join(sums) + "\n")
    with ZipFile(out) as z:
        if z.testzip() is not None:
            raise ValueError("Damaged output")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, default=EVIDENCE)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    txt = render(a.evidence)
    if a.bundle:
        bundle(a.evidence, a.bundle)
    elif a.check:
        if REPORT.read_text(encoding="utf8") != txt:
            raise SystemExit("Stale transfer report")
    else:
        REPORT.write_text(txt, encoding="utf8")
