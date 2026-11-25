"""Bind the native A/B to both binaries and correlate fresh acceptance outputs with v1."""

import argparse
import datetime
import hashlib
import json
import pathlib
import zipfile

from release import ROOT, digest


def generate(old_zip):
    reports = ROOT / "reports"

    def read(name):
        return json.loads((reports / (name + ".json")).read_text(encoding="utf8"))

    ab = read("native-ab")
    hashes = ab["binaries"]
    cases = ab["cases"]
    bundle_hash = digest(old_zip)
    if old_zip.with_suffix(".zip.sha256").read_text().split()[0] != bundle_hash:
        raise ValueError("baseline archive checksum mismatch")
    with zipfile.ZipFile(old_zip) as z:

        def before(name):
            return json.loads(z.read("reports/" + name + ".json"))

        baseline = {k: hashlib.sha256(z.read("build/" + k)).hexdigest() for k in hashes["old"]}
        old_ph = before("phantom")
        old_w = before("walnut-full")
        old_s = before("oversize")
    if (
        ab["pilot"]
        or not ab["passed"]
        or ab["pairs"] < 6
        or hashes["old"] != baseline
        or hashes["new"] != {k: digest(ROOT / "build" / k) for k in hashes["new"]}
    ):
        raise RuntimeError("missing passing comparison of these specific releases")
    ph = read("phantom")
    w = read("walnut-full")
    s = read("oversize")
    small = read("walnut")
    current = {f"analytic{n}": row["sha256"]["native"] for n, row in ph.items()}
    previous = {f"analytic{n}": row["sha256"]["native"] for n, row in old_ph.items()}
    current.update(walnut512=w["output_sha256"], stress1296=s["sha256"])
    previous.update(walnut512=old_w["output_sha256"], stress1296=old_s["sha256"])
    if current != previous or (
        s["native"]["peak_owned_bytes"] != old_s["native"]["peak_owned_bytes"]
    ):
        raise RuntimeError("large-volume output or stress allocation regression")
    for case, ref in [("phantom", "analytic512"), ("full-corrected", "walnut512")]:
        c = cases[case]
        if not c["passed"] or not all(
            m["output_sha256"] == current[ref]
            for m in [*c["warmup"].values(), *(m for pair in c["runs"] for m in pair)]
        ):
            raise RuntimeError("benchmark outputs do not match acceptance: " + case)
    repeat = dict(
        verified_on=datetime.datetime.now().astimezone().isoformat(),
        comparison="Full-volume v1 archive vs current full acceptance; same PC. Both 512^3 inputs additionally hashed in every A/B run.",
        baseline_zip_sha256=bundle_hash,
        binaries=hashes,
        baseline_outputs=previous,
        candidate_outputs=current,
        identical=True,
        release_demo_sha256=small["corrected_sha256"],
    )
    (reports / "run-repeat.json").write_text(json.dumps(repeat, indent=2) + "\n", encoding="utf8")
    text = [
        "# ConeFDK · 原生版本性能对照",
        "",
        f"同一台 Windows GPU；比较旧发布版与当前发布版，不是与 ASTRA 比速度。输出固定为 512³ FP32（512 MiB），显存预算 768 MiB，每批 16 角。两版都先各热身一次，再开独立进程交替跑 {ab['pairs']} 对，保留全部结果；没有清 OS/JIT 缓存，也没有锁定 GPU 时钟。两边均写同一目录同一目标、flush、原子替换。哈希在计时外。",
        "",
        "|输入|整进程中位数，旧 → 新（s）|耗时降幅|更快的配对|CLI 内部壁钟，旧 → 新（ms）|CUDA 反投影，旧 → 新（ms）|D2H，旧 → 新（ms）|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def pair(a, k, d):
        return f"{a[k]['old']:.{d}f} → {a[k]['new']:.{d}f}"

    for name, title in [("phantom", "解析体模，180 角"), ("full-corrected", "实测核桃，720 角")]:
        a = cases[name]["medians"]
        proc = a["process_seconds"]

        text.append(
            f"|{title}|{pair(a, 'process_seconds', 3)}|{proc['reduction_pct']:.1f}%|{proc['new_wins']}/{ab['pairs']}|{pair(a, 'wall_ms', 1)}|{pair(a, 'bp_ms', 1)}|{pair(a, 'd2h_ms', 1)}|"
        )
    text += [
        "",
        "进程时间从启动到退出，包含初始化、文件 I/O、同步和退出；CLI 内部壁钟不含退出清理；反投影及 D2H 是 CUDA Event 时间，不能混用。中位数不可逐项相加。两组均通过预设的端到端门槛：耗时中位数至少降低 5%，且至少 80% 的配对更快，内部壁钟也须分别满足。",
        "",
        "## 实现与代价",
        "",
        "- 旧内核每次采样都读四个独立的像素，再各自计算水平差值。新 CUDA kernel 在每个滤波后的角度批上预计算 `(左值, 右值−左值)`；反投影每层读上下两组 float2。不启用 fast-math，不用硬件近似双线性插值。虚拟左列与每角两个零元素维持原来的探测器外零延拓。",
        "- 编码表复用刚完成逆变换、已不再需要的复数 FFT 缓冲，不额外申请显存。`v*(u+1)+2 ≤ v*k` 对支持的 `u,v ≥ 2, k ≥ 2u` 成立，包含最紧的 2×2。每线程处理 16 个 Z 层，但分为原来的四层小组计算坐标，以保留非二进制体素间距的舍入次序。",
        "- 主机输入与输出暂存每块最多锁页 8 MiB。内存不足时退回普通页，也可用 `--host-memory pageable` 显式禁用。SSE2 检查 FP32 指数位，继续拒绝投影和重建中的非有限值。仍然校验 GPU 哨兵、落盘并原子发布。",
        "",
    ]
    a = cases["full-corrected"]
    filt = a["medians"]["filter_ms"]
    m = a["warmup"]["new"]
    text += [
        f"720 角实测：打表计入滤波阶段，该阶段中位数从 {filt['old']:.1f} 到 {filt['new']:.1f} ms。新旧设备缓冲峰值同为 {m['peak_owned_bytes'] / 2**20:.2f} MiB；新版本次另用 {m['host_pinned_bytes'] / 2**20:.2f} MiB 锁页**主机**内存。设备缓冲精确计数并非整个进程或驱动的显存。",
        "",
        "## 数值与复现",
        "",
        f"- 每份 A/B 输出都作完整 SHA-256，比对通过。完整 720 角核桃对 ASTRA 的全体积相对 L2 为 {100 * w['corrected_vs_astra']['rel_l2']:.5f}%。1296³ 压力体落盘 {s['output_bytes']:,} 字节且与旧包哈希相同；五个物理切面对 ASTRA 的验证并不等于全体积对照。",
        "- [逐轮数据与构建指纹](native-ab.json) · [跨版本输出指纹](run-repeat.json) · [完整验收](RESULTS.md) · [回归测试](native-tests.xml)。没有对未测试 GPU、临床或计量用途作认证。",
        "",
        "复现：先运行项目根目录的 `Run-Acceptance.ps1` 生成两套输入。把上一版解压到独立目录，执行 `python scripts/compare_native.py <旧版的build目录>`。生成交叉报告需保留上一版 ZIP 和旁边的 `.zip.sha256`：`python scripts/optimization_report.py <旧版ZIP>`。两条命令使用项目虚拟环境。",
    ]
    (reports / "OPTIMIZATION.md").write_text("\n".join(text) + "\n", encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline_zip", type=pathlib.Path)
    generate(p.parse_args().baseline_zip)
