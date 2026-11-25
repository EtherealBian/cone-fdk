"""Start from analytic detector rays and compare the fresh specimen bundle with a trusted run."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from compare_native import sha
from remount_loss import write_doc
from specimen_phantom import generate
from validate_roi import require
from validate_specimens import qualify


def replay(base, expected, evidence):
    require(
        not evidence.exists() and not expected.resolve().is_relative_to(base.resolve()),
        "use a fresh evidence file outside the reference, and new run folders",
    )
    expected_raw = expected.read_bytes()
    base.mkdir(parents=True, exist_ok=True)
    scans, work = base / "scans", base / "measurement"
    generate(scans)
    doc = qualify(scans, work)
    require(
        (work / "qualification.json").read_bytes() == expected_raw,
        "new detector-to-console run differs from trusted specimen bundle",
    )
    require(
        doc["console"]["html_sha256"] == sha(work / "console/index.html"),
        "unbound offline result",
    )
    write_doc(
        evidence,
        dict(
            protocol="Regenerated all five noisy cone acquisitions, recomputed both calibrated remounts per comparison, 27 global maps, truth matching, batch/resume and deterministic offline HTML. Exact qualification bytes compared to a prior run. Same toolchain, simulated specimens only.",
            projection_payloads=len(
                {p for c in doc["cases"].values() for p in c["projection_sha256"].values()}
            ),
            comparisons=len(doc["cases"]),
            pairs=doc["console"]["maps"],
            matched_qualification_sha256=sha(work / "qualification.json"),
            offline_html_sha256=doc["console"]["html_sha256"],
        ),
    )
    print("PASS: fresh analytic acquisitions through identical offline result")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected", required=True, type=Path)
    p.add_argument("--evidence", required=True, type=Path)
    p.add_argument("--keep", type=Path)
    a = p.parse_args()
    with TemporaryDirectory() as t:
        replay(a.keep or Path(t), a.expected, a.evidence)
