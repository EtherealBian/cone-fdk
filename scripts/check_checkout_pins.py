"""Verify that the checked-out sources and published offline artifacts have their attested bytes."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(root):
    reports, scripts = root / "reports", root / "scripts"
    edge = json.loads((reports / "edge-qualification.json").read_bytes())
    archive = reports / "edge-review-example.zip"
    if digest(archive) != edge["evidence_sha256"][archive.name]:
        raise ValueError("edge example archive mismatch")
    with zipfile.ZipFile(archive) as z:
        raw = z.read("manifest.json")
    if hashlib.sha256(raw).hexdigest() != edge["cases"]["noisy-change"]["review_manifest_sha256"]:
        raise ValueError("archived edge manifest mismatch")
    specimens = json.loads((reports / "specimen-qualification.json").read_bytes())
    replay = json.loads((reports / "specimen-replay.json").read_bytes())
    checks = {
        scripts / k: v
        for d in (json.loads(raw)["software_sha256"], specimens["sources_sha256"])
        for k, v in d.items()
    }
    checks.update(
        {
            reports / "specimen-console.html": specimens["console"]["html_sha256"],
            reports / "specimen-qualification.json": replay["matched_qualification_sha256"],
        }
    )
    if replay["offline_html_sha256"] != specimens["console"]["html_sha256"]:
        raise ValueError("specimen evidence does not bind console")
    browser = json.loads((reports / "region-console-browser.json").read_bytes())
    checks.update(
        {
            reports / "region-console.html": browser["html_sha256"],
            reports / "region-console.png": browser["screenshot_sha256"],
            reports / "region-qualification.json": browser["native_qualification_sha256"],
        }
    )
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError("checkout digest mismatch: " + str(path.relative_to(root)))
    print(f"PASS: {len(checks)} attested checkout files, including legacy Windows source bytes")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    check(p.parse_args().root)
