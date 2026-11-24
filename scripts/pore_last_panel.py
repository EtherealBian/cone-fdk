"""Extract only the last panel of the pinned paper for a local numerical audit."""

import argparse
import hashlib
from pathlib import Path

from pypdf import PdfReader

PDF = "6750add7c027cd08c158faa357aaa90a102a081bacae22b82cbbfbe794b92450"


def extract(f, out):
    if hashlib.sha256(f.read_bytes()).hexdigest() != PDF:
        raise ValueError("Wrong PDF")
    p = PdfReader(f).pages[5].images["/Im15"].image.convert("L")
    if p.size != (1800, 1013) or out.exists():
        raise ValueError("Wrong image or occupied output path")
    p.crop((1352, 466, 1800, 914)).save(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paper", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    extract(a.paper, a.out)
