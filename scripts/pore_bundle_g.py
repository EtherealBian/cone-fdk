"""Create self-contained G/H development evidence packages; preserve sealed field bytes."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from pore_bundle_f import ROOT, sha


def bundle(formal, pred, out, stage="G"):
    if stage not in ("G", "H"):
        raise ValueError("Unsupported development stage")
    suffix_id = stage.lower()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose an empty output directory")
    evidence = ROOT / f"pores/evidence/candidate-{suffix_id}"
    manifest = json.loads((pred / "predictions.json").read_text())
    if manifest != json.loads((evidence / "predictions.json").read_text()) or set(
        manifest["specimens"]
    ) != {"2", "3", "5"}:
        raise ValueError("Different sealed manifest")
    if stage == "H":
        from pore_shared_score import shared_digest

        shared_digest(manifest)
    for i in (2, 3, 5):
        s = f"sample{i}"
        r = manifest["specimens"][str(i)]
        zpath = out / f"ConeFDK-{stage}-{s}.zip"
        report = json.loads((formal / (s + ".json")).read_text())
        fields = {
            s + ".f32": r["field" if stage == "G" else "logits"],
            s + "-weights.npz": r["weights"],
        }
        if (
            report["seal"] != r
            or sha(formal / (s + "-review.html")) != report["review_sha256"]
            or any(sha(pred / k) != v for k, v in fields.items())
        ):
            raise ValueError("Prediction or viewer changed")
        with ZipFile(zpath, "x", compression=ZIP_DEFLATED, compresslevel=5) as z:
            for suffix in (
                ".json",
                "-pores.csv",
                "-matches.csv",
                "-cases.csv",
                "-size-bands.csv",
                "-overlap.csv",
                *[
                    f"-paired-{prior}-{suffix_id}.csv"
                    for prior in ("ef" if stage == "G" else "efg")
                ],
                "-cases.html",
                "-review.html",
            ):
                p = formal / (s + suffix)
                if p.suffix != ".html" and sha(p) != sha(evidence / p.name):
                    raise ValueError("Different public report")
                z.write(p, "formal/" + p.name)
            for k in [*fields, "predictions.json"]:
                z.write(pred / k, "predictions/" + k)
            z.write(evidence / "raw-weight-replay.json", f"audit/{stage}-raw-weight-replay.json")
            z.writestr(
                "READ-ME.txt",
                f"Extract the three packages in one directory. Open formal/{s}-cases.html or -review.html offline. See pores/RESULTS-{stage}.md in measurement-evidence-{5 if stage == 'G' else 6} for source-archive replay. Development only; passed all eleven on this specimen: {report['final_passed']}. A development run cannot certify the method. XCT source and digital labels: Kim et al., NIST DOI 10.18434/M32162. Derived display slices and predictions, 2026-11-24. Not physical certification.\n",
            )
        with ZipFile(zpath) as z:
            if z.testzip():
                raise ValueError("Invalid ZIP")
        print(zpath, zpath.stat().st_size, flush=True)
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{sha(p)}  {p.name}\n" for p in sorted(out.glob("*.zip"))), encoding="ascii"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("formal", "pred", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    ap.add_argument("--stage", choices=["G", "H"], default="G")
    a = ap.parse_args()
    bundle(a.formal, a.pred, a.out, a.stage)
