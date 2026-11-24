"""Trace the specimen-5 matched-volume outlier to its contributing reference IDs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pore_transfer import ROOT, Z, label
from pore_transfer_replay import read_npz
from pore_transfer_report import capacity
from scipy import ndimage as ndi

sys.path.insert(0, str(ROOT / "pores"))
from validate import stack, verified  # noqa: E402


def run(folder, cache, out):
    record = json.loads((folder / "sample5.json").read_text(encoding="utf8"))
    rid = capacity(record)[-1][1]  # Worst ideal residual, explicitly a post-hoc witness
    row = record["base"]["references"][str(rid)]
    pid = row["pred"]
    _, base, _, truth = read_npz(folder / "sample5-review.npz")
    lab, rc, whole, _ = label(truth)
    pred = label(base)[0] == pid
    hits = np.bincount(lab[pred], minlength=len(rc))
    other = max((int(hits[k]), k) for k in np.flatnonzero(hits[1:]) + 1 if k != rid)[1]
    both = [rid, int(other)]
    planes = []
    for z in np.flatnonzero(pred.any(axis=(1, 2))):
        cc, num = ndi.label(pred[z])
        t = lab[z]
        for idx in range(1, num + 1):
            parts = [int(np.sum((cc == idx) & (t == r))) for r in both]
            if min(parts) > 0:
                planes.append((min(parts), -int(z), parts))
    _, negz, parts = max(planes)
    z = -negz
    yy, xx = np.nonzero(pred[z] | np.isin(lab[z], both))
    cy, cx = int((yy.min() + yy.max()) // 2), int((xx.min() + xx.max()) // 2)
    half = max(12, int(max(np.ptp(yy), np.ptp(xx))) // 2 + 5)
    y0, y1 = max(0, cy - half), min(256, cy + half + 1)
    x0, x1 = max(0, cx - half), min(256, cx + half + 1)
    src, source = verified(cache, 5, "raw", False)
    raw = stack(src, 5, "raw", *Z)
    lo, hi = np.percentile(raw, [1, 99])
    gray = np.clip(
        (raw[z, y0:y1, x0:x1].astype(float) - lo) * 255 / max(1, hi - lo), 0, 255
    ).astype("u1")
    palette = [(93, 196, 255), (255, 199, 93), (92, 224, 168), (237, 135, 243)]
    t = lab[z, y0:y1, x0:x1]
    p = pred[z, y0:y1, x0:x1]

    def panel(layers):
        img = np.repeat(gray[:, :, None], 3, axis=2)
        for m, color in layers:
            img[m] = np.rint(img[m] * 0.25 + np.array(color) * 0.75).astype("u1")
        return Image.fromarray(img).resize((248, 248), Image.Resampling.NEAREST)

    case = dict(
        specimen=5,
        reference=rid,
        pred=pid,
        predicted_voxels=int(pred.sum()),
        reference_voxels=int(rc[rid]),
        missed_reference_voxels=int(rc[rid] - hits[rid]),
        reference_background_in_prediction=int(hits[0]),
        overlaps=[
            dict(
                reference=int(k),
                overlap_voxels=int(hits[k]),
                reference_voxels=int(rc[k]),
                complete=bool(whole[k]),
            )
            for k in np.flatnonzero(hits[1:]) + 1
        ],
        source_z=z + Z[0],
        source_xy_halfopen=[360 + x0, 376 + y0, 360 + x1, 376 + y1],
        same_four_connected_prediction_section_hits=dict(zip(both, parts, strict=True)),
        raw_sha256=source["sha256"],
    )
    canvas = Image.new("RGB", (528, 600), "#0a1b20")
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    d.text(
        (12, 8),
        f"SPECIMEN 05     /     SOURCE SLICE {z + Z[0]:04}     /     XY",
        font=font,
        fill="#b2d6ce",
    )
    panels = [
        ("01   Raw XCT", []),
        (f"02   Reference {rid} / {other}", [(t == rid, palette[0]), (t == other, palette[1])]),
        (f"03   E component {pid}", [(p, palette[2])]),
        (
            "04   Inside E: overlap / other / extra",
            [
                (p & (t == rid), palette[0]),
                (p & (t == other), palette[1]),
                (p & (t == 0), palette[3]),
            ],
        ),
    ]
    for j, (title, layers) in enumerate(panels):
        x, y = 12 + 256 * (j % 2), 38 + 279 * (j // 2)
        d.text((x, y), title, font=ImageFont.load_default(size=14), fill="white")
        canvas.paste(panel(layers), (x, y + 24))
    out.mkdir(exist_ok=True, parents=True)
    canvas.save(out / "sample5-merge.png")
    (out / "sample5-merge.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf8")
    return case


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(run(a.directory, a.cache, a.out))
