"""Bundle sealed F fields and split controls without changing the scored reports."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def bundle(formal, pred, split, out):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose an empty bundle directory")
    for i in (2, 3, 5):
        s = f"sample{i}"
        dest = out / f"ConeFDK-F-split-{s}.zip"
        with ZipFile(dest, "x", compression=ZIP_DEFLATED, compresslevel=4) as z:
            for fol, source, extra in [
                ("formal", formal, []),
                ("predictions", pred, ["predictions.json"]),
                ("split", split, ["decision.json"]),
            ]:
                paths = sorted(source.glob(s + "-*")) + [
                    p for p in (source / (s + ".json"), source / (s + ".f32")) if p.is_file()
                ]
                if not paths:
                    raise ValueError("Missing specimen " + s)
                for p in paths + [source / k for k in extra]:
                    z.write(p, fol + "/" + p.name)
            z.write(ROOT / "pores/evidence/split-pilot/seal.json", "split/seal.json")
            z.writestr(
                "READ-ME.txt",
                "Research only: F did not pass FINAL-GATE. Extract all three specimen packages to a common directory. Open formal/"
                + s
                + "-cases.html or -review.html for an offline slice-by-slice comparison. See pores/RESULTS-F.md in the measurement-evidence-4 source tag for numerical replay. Derived from NIST Kim et al., DOI 10.18434/M32162: contrast-windowed review images, predictions and digital-reference comparisons, 2026-11-24. Not physical metrology certification.\n",
            )
        with ZipFile(dest) as z:
            if z.testzip():
                raise ValueError("Bad ZIP")
        print(dest.name, dest.stat().st_size, flush=True)
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{sha(p)}  {p.name}\n" for p in sorted(out.glob("*.zip"))), encoding="ascii"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("formal", "pred", "split", "out"):
        ap.add_argument("--" + k, required=True, type=Path)
    a = ap.parse_args()
    bundle(a.formal, a.pred, a.split, a.out)
