"""Build one network-free, four-panel source slice viewer per transfer fold."""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pore_transfer import DEV, ROOT, Z, label
from pore_transfer_replay import read_npz

sys.path.insert(0, str(ROOT / "pores"))
from validate import stack, verified  # noqa: E402


def pack_plane(gray, truth, base, arm):
    flags = truth.astype("u1") + 2 * base.astype("u1") + 4 * arm.astype("u1")
    rgb = np.stack((gray, flags << 5, np.zeros_like(gray)), axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def locator(lab, boxes, k):
    box = boxes[k - 1]
    peak = int(np.argmax((lab[box] == k).sum(axis=(1, 2)))) + box[0].start
    return dict(box=[v for s in box for v in (s.start, s.stop)], peak=peak)


def build(folder, cache, i):
    rec = json.loads((folder / f"sample{i}.json").read_text())
    _, base, arm, truth = read_npz(folder / f"sample{i}-review.npz")
    raw, source = verified(cache, i, "raw", False)
    if source["sha256"] != rec["sealed"]["raw"]:
        raise ValueError("Different raw TIFF archive")
    a = stack(raw, i, "raw", *Z)
    lo, hi = np.percentile(a, [1, 99])
    grey = np.clip((a.astype("f8") - lo) * 255 / max(1, hi - lo), 0, 255).astype("u1")
    layers = [pack_plane(*quad) for quad in zip(grey, truth, base, arm, strict=True)]
    cases = [
        dict(
            title="整幅切片",
            note="选择一个孔定位到其最大横截面，或拖动滑块查看整段 96 层。",
            peak=48,
            box=None,
        )
    ]
    lr, _, _, boxes = label(truth)
    for key, row in rec["base"]["references"].items():
        other = rec["arm"]["references"][key]
        n = row["reference_voxels"]
        b = row["voxels"]
        a = other["voxels"]

        def text(v):
            return str(v) + " 体素" if v is not None else "未匹配"

        cases.append(
            dict(
                title=f"参考 #{key} · {n} 体素",
                note=f"参考 {n} 体素；候选 E：{text(b)}；跨件修正：{text(a)}。",
                **locator(lr, boxes, int(key)),
            )
        )
    for key, mask, arm_name in [("base", base, "候选 E"), ("arm", arm, "跨件修正")]:
        lp, sizes, whole, bs = label(mask)
        used = {r["pred"] for r in rec[key]["references"].values() if r["pred"] is not None}
        for pid in np.flatnonzero(whole & (sizes >= 64)):
            if pid not in used:
                cases.append(
                    dict(
                        title=f"{arm_name}未匹配预测 #{pid}",
                        note=f"预测体积 {sizes[pid]} 体素；没有符合资格的一对一匹配，不计为零误差。",
                        **locator(lp, bs, int(pid)),
                    )
                )
    payload = dict(i=i, window=[float(lo), float(hi)], layers=layers, cases=cases)
    tmpl = (ROOT / "scripts/transfer_review.html").read_text(encoding="utf8")
    (folder / f"sample{i}-slices.html").write_text(
        tmpl.replace("PAYLOAD", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        encoding="utf8",
    )
    print(i, len(cases) - 1, "objects available in offline viewer", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    for i in DEV:
        build(a.directory, a.cache, i)
