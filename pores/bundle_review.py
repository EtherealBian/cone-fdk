"""Package a self-contained numerical review for sharing without Python, GPU or network."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def write_bundle(src, dest, specimen):
    stem = f"sample{specimen}"
    report = json.loads((src / (stem + ".json")).read_text(encoding="utf-8"))
    data = {}
    for suffix in (
        ".json",
        "-pores.csv",
        "-matches.csv",
        "-review.html",
        "-cases.csv",
        "-cases.html",
    ):
        name = stem + suffix
        data[name] = (src / name).read_bytes()

    def sha(content):
        return hashlib.sha256(content).hexdigest()

    if report["review_sha256"] != sha(data[stem + "-review.html"]) or report[
        "table_sha256"
    ] != sha(data[stem + "-pores.csv"]):
        raise ValueError("Refusing to bundle incomplete or mismatched evidence")
    data["READ-ME.txt"] = (
        f"ConeFDK / experimental numerical evidence / {report['method']}\n"
        f"Specimen {specimen} ({report['role']}); passed all numerical gates: {report['final_passed']}.\n"
        f"Unzip and open {stem}-review.html locally. Select a CSV ID or slice. In Compare: white = agreement on void, orange = prediction only, cyan = reference only.\n"
        f"Open {stem}-cases.html for missed references and the largest volume discrepancies, with raw/overlay crops and exact source coordinates.\n"
        "Open the JSON for exact source hashes and metrics, and the two CSVs for predicted components and matched pairs.\n"
        "Data: Felix H. Kim et al., NIST, https://doi.org/10.18434/M32162 . Reconstructed TIFFs cropped and annotated; no endorsement or physical dimensional certification.\n"
    ).encode("utf-8")
    data["SHA256SUMS.txt"] = "".join(f"{sha(data[n])}  {n}\n" for n in sorted(data)).encode(
        "utf-8"
    )
    dest.parent.mkdir(exist_ok=True, parents=True)
    with ZipFile(dest, "w") as z:
        for n in sorted(data):
            item = ZipInfo(n, date_time=(2026, 11, 23, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            item.create_system = 0
            z.writestr(item, data[n])
    with ZipFile(dest) as z:
        if z.testzip() or any(z.read(n) != v for n, v in data.items()):
            raise ValueError("ZIP round-trip failed")
    return sha(dest.read_bytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("specimen", type=int)
    a = ap.parse_args()
    print(write_bundle(a.source, a.out, a.specimen))
