"""Restore the published packed noisy-change example using the original, byte-identical wall files."""

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


def checked(flag, reason):
    if not flag:
        raise ValueError(reason)


def bound(path, cap, expected):
    with path.open("rb") as f:
        data = f.read(cap + 1)
    checked(
        len(data) <= cap and hashlib.sha256(data).hexdigest() == expected,
        f"invalid size or digest: {path.name}",
    )
    return data


def restore(reports, job, qualification_sha256):
    reports, job = reports.resolve(), job.resolve()
    checked(
        isinstance(qualification_sha256, str)
        and re.fullmatch("[0-9a-f]{64}", qualification_sha256),
        "external qualification SHA-256 required",
    )
    checked(not job.is_relative_to(reports), "write outside the evidence directory")
    if job.exists():
        raise FileExistsError(job)
    qualification = json.loads(
        bound(reports / "packed-edge-qualification.json", 1 << 20, qualification_sha256)
    )
    case = qualification["cases"]["noisy-change"]
    overlay = bound(
        reports / "packed-edge-example-manifest.json", 1 << 20, case["packed_edge_sha256"]
    )
    doc = json.loads(overlay)
    checked(
        doc["format"] == "CFDK-edge-review-v1"
        and doc["backend"] == "witness-wall-pack-v1"
        and doc["inputs_sha256"]["manifest.json"] == case["parent_sha256"]
        and doc["inputs_sha256"]["wall-pack.exe"]
        == qualification["qualified_build"]["executable_sha256"]
        and doc["outputs_sha256"] == case["matched_outputs_sha256"],
        "overlay provenance",
    )
    h, w = doc["shape_ha"]
    cells = h * w
    levels = doc["levels_per_mm"]
    checked(
        type(w) is int
        and type(h) is int
        and 3 <= w <= 4096
        and 1 <= h <= 4096
        and cells <= 1 << 20
        and 2 <= len(levels) <= 16,
        "example size limit",
    )
    lengths = {
        f"{r}-{i:02}.wall": 32 * cells
        for r in ("reference", "inspected")
        for i in range(len(levels))
    }
    lengths.update({"class.u8": cells, "min-loss.f64": 8 * cells, "max-loss.f64": 8 * cells})
    checked(
        set(doc["outputs_sha256"]) == {"material.grid", *lengths}
        and sum(lengths.values()) <= 128 << 20,
        "unexpected example layout",
    )
    archive = reports / "edge-review-example.zip"
    # Never extract arbitrary archive paths. Reuse only fixed member names into a new staging folder.
    checked(archive.stat().st_size <= 132 << 20, "archive size limit")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        checked(
            len(names) == len(set(names))
            and set(names) == {"manifest.json", *doc["outputs_sha256"]},
            "unexpected archive member",
        )
        checked(z.getinfo("manifest.json").file_size <= 1 << 20, "archive manifest too large")
        raw = z.read("manifest.json")
        checked(
            hashlib.sha256(raw).hexdigest() == case["single_edge_sha256"],
            "single-edge manifest mismatch",
        )
        checked(json.loads(raw)["outputs_sha256"] == doc["outputs_sha256"], "different payloads")
        job.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=job.parent) as tmp:
            stage = Path(tmp) / "review"
            stage.mkdir()
            for n, pin in doc["outputs_sha256"].items():
                size = z.getinfo(n).file_size
                checked(
                    size == lengths[n] if n in lengths else 0 < size <= 4096,
                    "invalid member length",
                )
                with z.open(n) as src, (stage / n).open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                with (stage / n).open("rb") as f:
                    checked(
                        hashlib.file_digest(f, "sha256").hexdigest() == pin,
                        "payload digest mismatch: " + n,
                    )
            (stage / "manifest.json").write_bytes(overlay)
            stage.rename(job)
    return case["packed_edge_sha256"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path)
    p.add_argument("job", type=Path)
    p.add_argument("--qualification-sha256", required=True)
    try:
        digest = restore(**vars(p.parse_args()))
    except (ValueError, OSError, zipfile.BadZipFile, KeyError, TypeError) as err:
        p.exit(1, f"offline sample restore failed: {err}\n")
    print("Restored packed edge manifest SHA-256:", digest)
