"""Post-hoc label census: quantify calibration support and replay the shell control's denominators."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
from validate import stack, verified  # noqa: E402


def census(vol):
    labels, n = ndi.label(vol)
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    complete = [
        int(counts[i])
        for i, box in enumerate(ndi.find_objects(labels), start=1)
        if box is not None
        and all(s.start > 0 and s.stop < dim for s, dim in zip(box, vol.shape, strict=True))
    ]
    return {
        "complete_32_63": sum(32 <= v < 64 for v in complete),
        "complete_64_127": sum(64 <= v < 128 for v in complete),
        "complete_ge128": sum(v >= 128 for v in complete),
        "clipped_ge64": int(np.sum(counts[1:] >= 64)) - sum(v >= 64 for v in complete),
    }


def run(cache, records):
    result = {}
    for i in (2, 3, 5):
        path, src = verified(cache, i, "segmented", False)
        r = json.loads((records / f"sample{i}.json").read_text(encoding="utf8"))
        if (
            r["stage"] != "calibration only"
            or r["specimen"] != i
            or src["sha256"] != r["reference_sha256"]
        ):
            raise ValueError("Wrong source")
        y = stack(path, i, "segmented", 160, 184)
        whole = census(y)
        sides = {}
        for fold, bounds in {"L": (0, 96), "R": (160, 256)}.items():
            a, b = bounds
            f = census(y[:, :, a:b])
            sides[fold] = f
            for arm, row in r["checked_folds"][fold].items():
                if (
                    f["complete_64_127"] + f["complete_ge128"] != row["eligible_reference"]
                    or f["complete_64_127"] != len(row["small_references"])
                    or f["clipped_ge64"] != row["clipped_reference_ge64"]
                ):
                    raise ValueError(f"Label census disagrees for {i}/{fold}/{arm}")
        result[str(i)] = {"segmented_sha256": src["sha256"], "whole": whole, "folds": sides}
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = json.dumps(run(a.cache, a.records), indent=2) + "\n"
    if a.check:
        if a.out.read_text(encoding="utf8") != text:
            raise SystemExit("Stale calibration census")
    else:
        a.out.write_text(text, encoding="utf8")
