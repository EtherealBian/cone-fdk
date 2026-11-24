"""Export only grayscale renderings for the sample-2 source locator, never a reference mask."""

import argparse
import hashlib
from pathlib import Path

from pypdf import PdfReader

PDF = "6750add7c027cd08c158faa357aaa90a102a081bacae22b82cbbfbe794b92450"


def extract(pdf, out, include_nlm=False):
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != PDF:
        raise ValueError("Wrong paper")
    page = PdfReader(pdf).pages[5]
    v = page.images["/Im15"]
    a = v.image.convert("L")
    if a.size != (1800, 1013):
        raise ValueError("Unexpected sample 2 rendering")
    out.mkdir(parents=True, exist_ok=False)
    for name, box in [
        ("overview", (2, 5, 452, 466)),
        ("raw", (2, 466, 450, 914)),
        ("median", (452, 466, 900, 914)),
    ]:
        a.crop(box).save(out / f"{name}.png")
    if include_nlm:
        a.crop((902, 466, 1350, 914)).save(out / "nlm.png")
    print("Exported local grayscale crops from PDF page 6.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-nlm", action="store_true")
    a = ap.parse_args()
    extract(a.pdf, a.out, a.include_nlm)
