"""Calibration-only pilot: is the first 3x3x3 median's footprint driving voxel errors?"""

import argparse
import json
import sys
from pathlib import Path

from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pores"))
from denoise import Denoise  # noqa: E402
from local_model import SETTINGS, context_stack, local_fields, segment  # noqa: E402
from pore_check import ROOT  # noqa: E402
from validate import score, stack, verified  # noqa: E402


def pilot(i, cache):
    raw, _ = verified(cache, i, "raw", False)
    seg, _ = verified(cache, i, "segmented", False)
    slope, _, ct = SETTINGS[i]
    # Never load evaluation labels or images. Eight unused planes separate the two blocks.
    windows = ((160, 168), (176, 184))
    raw_blocks = [context_stack(raw, i, a, b) for a, b in windows]
    ref = [stack(seg, i, "segmented", a, b) for a, b in windows]
    result = {}
    with Denoise(ROOT / "build/pores/nlm.ptx") as gpu:
        supports = {
            name: ndi.generate_binary_structure(3, k)
            for name, k in [("cube27", 3), ("edges19", 2), ("faces7", 1)]
        }
        for name, foot in supports.items():
            imgs = [
                ndi.median_filter(im, footprint=foot, mode="nearest")[1:-1] for im in raw_blocks
            ]
            best = None
            for band in (14, 22, 30, 42):
                features = local_fields(gpu(imgs[0], band / slope), i)
                for offset in (-2, 0, 2):
                    for bias in (-1, 0, 1):
                        v = score(segment(features, ct + offset, bias), ref[0])
                        rank = (v["dice"], -abs(offset), -abs(bias), -band, -offset, -bias)
                        if best is None or rank > best[0]:
                            best = rank, (band, offset, bias), v
            _, (band, offset, bias), training = best
            final = segment(local_fields(gpu(imgs[1], band / slope), i), ct + offset, bias)
            result[name] = dict(
                voxels_in_support=int(foot.sum()),
                band=band,
                contrast_offset=offset,
                threshold_bias=bias,
                training=training,
                calibration_check=score(final, ref[1]),
            )
            print(i, name, result[name]["calibration_check"]["dice"], flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = {str(i): pilot(i, args.cache) for i in (2, 3, 5)}
    promote = [
        key
        for key in ("edges19", "faces7")
        if all(
            row[key]["calibration_check"]["dice"] - row["cube27"]["calibration_check"]["dice"]
            >= 0.005
            for row in rows.values()
        )
    ]
    record = dict(
        scope="calibration-only pilot of direct 2-D NLM/Bernsen; not the C/D classifier or an acceptance run",
        fit_z=[160, 168],
        check_z=[176, 184],
        required_per_scan_dice_gain=0.005,
        scans=rows,
        promote=promote,
    )
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print("Support changes meeting prespecified pilot criterion:", promote)
