"""Independent native-center replay of the registered figure check; never write image data."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "pores/evidence"
PDF_SHA = "6750add7c027cd08c158faa357aaa90a102a081bacae22b82cbbfbe794b92450"


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def axis_centers(start, stop, origin, scale):
    if not 0 < scale < 1:
        raise ValueError("Expected an enlarged panel")
    for dest in range(start, stop):
        src = math.floor(origin + dest * scale + 0.5)
        if math.floor((src - origin) / scale + 0.5) == dest:
            yield src, dest


def rates(cells, target, test):
    tn, fp, fn, tp = [
        sum(n for j, n in enumerate(cells) if bool(j & target) == a and bool(j & test) == b)
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]
    ]
    return dict(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        dice=2 * tp / (2 * tp + fp + fn),
        precision=tp / (tp + fp),
        recall=tp / (tp + fn),
    )


def fixed_bytes(volume):
    # Valid cube median, followed by valid displacements. The box integral sums
    # are exact in FP64 (25 squared uint16 differences); no CUDA code is shared.
    h, w = volume.shape[1:]
    med = np.median(
        np.stack(
            [
                im[y : y + h - 2, x : x + w - 2]
                for im in volume
                for y in range(3)
                for x in range(3)
            ]
        ),
        axis=0,
    )
    h, w = med.shape
    a = med[10 : h - 10, 10 : w - 10]
    num = np.zeros_like(a[2:-2, 2:-2])
    den = np.zeros_like(num)
    for dy in range(21):
        for dx in range(21):
            b = med[dy : h - 20 + dy, dx : w - 20 + dx]
            p = np.pad((a - b) ** 2, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
            d = p[5:, 5:] - p[:-5, 5:] - p[5:, :-5] + p[:-5, :-5]
            weight = np.exp(-d / (25 * (30 / 0.0085) ** 2))
            num += weight * b[2:-2, 2:-2]
            den += weight
    # First output is source (13,13) relative to the raw crop.
    return np.clip(np.floor((num / den - 25000) * 0.0085 + 0.5), 0, 255).astype("u1")


def region_stats(cells, mid):
    ap = rates(cells, 4, 1)
    ok = all(
        min(sum(n for j, n in enumerate(cells) if bool(j & bit) == value) for value in (0, 1))
        >= 32
        for bit in (4, 1)
    )
    return dict(
        count=sum(cells),
        archive_fixed_printed=cells,
        intermediate_tones=mid,
        archive_reference_printed=ap,
        archive_reference_fixed=rates(cells, 4, 2),
        printed_reference_fixed=rates(cells, 1, 2),
        sufficient=ok,
        concordant=ok and ap["dice"] >= 0.97 and min(ap["precision"], ap["recall"]) >= 0.95,
    )


def replay(cache, pdf):
    from pypdf import PdfReader

    e = ROOT / "figure-binary"
    s = json.loads((e / "source.json").read_bytes())
    ref = json.loads((e / "check.json").read_bytes())
    r = ROOT / "figure-locator/registration.json"
    reg = json.loads(r.read_bytes())
    raw = cache / "raw.zip"
    seg = cache / "seg.zip"
    if (
        file_sha(pdf) != PDF_SHA
        or file_sha(seg) != s["segmented"]
        or file_sha(raw) != s["raw"]
        or file_sha(r) != s["registration"]
        or file_sha(e / "native.npz") != s["file"]
        or reg["z"] != 447
    ):
        raise ValueError("Different source")
    page = PdfReader(pdf).pages[5].images["/Im15"].image.convert("L")
    assert page.size == (1800, 1013)
    with ZipFile(seg) as z:
        label = np.array(
            Image.open(io.BytesIO(z.read("set1sample2segmented/set1sample2segmented_0447.tif")))
        )
    box = s["raw_support"]
    with ZipFile(raw) as z:
        vol = np.stack(
            [
                np.asarray(
                    Image.open(
                        io.BytesIO(z.read(f"set1sample2raw/set1sample2raw_{i:04d}.tif"))
                    ).crop(tuple(box))
                )
                for i in (446, 447, 448)
            ]
        )
    gray = fixed_bytes(vol)
    ox, oy, scale = reg["source_xy0_scale"]
    reports = {}
    offsets = [(dy, dx) for dy in range(-5, 6) for dx in range(-5, 6) if dx * dx + dy * dy <= 26]
    assert gray.shape == (box[3] - box[1] - 26, box[2] - box[0] - 26)
    with np.load(e / "native.npz", allow_pickle=False) as pack:
        for split, (lo, hi) in {"fit": (112, 216), "check": (232, 428)}.items():
            cells = [0] * 8
            mid = [0] * 8
            coords = []
            errors = [0, 0]
            fixed = pack[split + "_fixed"]
            archived = pack[split + "_archive"]
            for y, v in axis_centers(80, 424, oy, scale):
                for x, u in axis_centers(lo, hi, ox, scale):
                    j = len(coords)
                    coords.append((x, y, u, v))
                    yy = y - 13 - box[1]
                    xx = x - 13 - box[0]
                    assert (
                        min(yy, xx) >= 5
                        and yy + 5 < gray.shape[0]
                        and xx + 5 < gray.shape[1]
                        and label[y, x] in (0, 255)
                    )
                    disk = [int(gray[yy + dy, xx + dx]) for dy, dx in offsets]
                    low, high = min(disk), max(disk)
                    threshold = (low + high) // 2
                    pred = threshold < 128 if high - low < 15 else int(gray[yy, xx]) < threshold
                    a = bool(label[y, x] == 0)
                    b = bool(fixed[j])
                    tone = page.getpixel((1352 + u, 466 + v))
                    key = 4 * a + 2 * b + (tone < 128)
                    cells[key] += 1
                    mid[key] += int(16 <= tone <= 239)
                    errors[0] += int(a != archived[j])
                    errors[1] += int(pred != b)
            xy = np.array(coords, dtype="<i4")
            coordsha = hashlib.sha256(xy.tobytes()).hexdigest()
            assert (
                np.array_equal(xy, pack[split + "_xyuv"])
                and coordsha == s["arrays"][split + "_xyuv"]
                and errors[0] == 0
            )
            info = region_stats(cells, mid)
            assert ref["partitions"][split] == info
            reports[split] = dict(
                xyuv_sha256=coordsha,
                archive_bit_differences=errors[0],
                cpu_filter_bit_differences=errors[1],
                **info,
            )
    assert ref["concordant"] == all(v["concordant"] for v in reports.values())
    return dict(
        pdf_sha256=PDF_SHA,
        segmented=s["segmented"],
        raw=s["raw"],
        source_sha256=file_sha(e / "source.json"),
        score_sha256=file_sha(e / "check.json"),
        partitions=reports,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    result = replay(args.cache, args.pdf)
    with args.out.open("x", encoding="utf8", newline="\n") as fp:
        fp.write(json.dumps(result, indent=2) + "\n")
