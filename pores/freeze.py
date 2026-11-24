"""Freeze a successful development method before running either registered holdout."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from audit import replay, require
from pore_check import ROOT
from validate import sha

DEV = (2, 3, 5)
HOLDOUT = (4, 6)
NAMES = {
    "A": "calibrated NLM residual classifier A",
    "B": "blockwise NLM classifier B",
    "C": "multiscale boundary ensemble C",
    "D": "axial neck and boundary estimator D",
}


def identity(model):
    """The actual Python sources, compiled artifacts, settings and numerical environment."""
    path = ROOT / "pores"
    files = [p for p in path.glob("*.py") if not p.name.startswith("test_")]
    files += [path / n for n in ("sources.json", "FINAL-GATE.md", "PROTOCOL.md", "review.html")]
    hashes = {
        p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files
    }
    for name in (
        "cone-pores.exe",
        "pores.ptx",
        {"A": "nlm.ptx", "B": "patches.ptx", "C": "multiscale.ptx", "D": "adaptive.ptx"}[model],
    ):
        hashes[name] = sha(ROOT / "build" / "pores" / name)
    packages = {
        n: version(n) for n in ("numpy", "scipy", "scikit-learn", "pillow", "threadpoolctl")
    }
    return {
        "model": model,
        "files": hashes,
        "python": platform.python_version(),
        "packages": packages,
    }


def token(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def certify_development(directory, cache, model):
    require(__debug__, "Do not run a numerical acceptance audit with -O")
    locked = identity(model)
    reports = {}
    for i in DEV:
        method, gates = replay(directory, i, cache)
        failed = [name for name, ok in gates.items() if not ok]
        require(
            method == NAMES[model] and gates and not failed,
            f"Development specimen {i} cannot unlock holdouts: {failed or 'method mismatch'}",
        )
        path = directory / f"sample{i}.json"
        report = json.loads(path.read_text(encoding="utf8"))
        require(
            report.get("pipeline_id") == token(locked),
            f"Specimen {i}: rerun this frozen pipeline on development data first",
        )
        reports[str(i)] = sha(path)
    return {
        "schema": 1,
        "holdout": list(HOLDOUT),
        "pipeline": locked,
        "development_reports": reports,
    }


def checked_manifest(path, directory, cache, model):
    saved = json.loads(path.read_text(encoding="utf8"))
    require(
        saved == certify_development(directory, cache, model),
        "Manifest differs from current build or replayed development evidence",
    )
    return token(saved)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "directory", type=Path, help="Directory containing the three full development reviews"
    )
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", choices=list(NAMES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = certify_development(a.directory, a.cache, a.model)
    with a.out.open("x", encoding="utf8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        "Development gates passed. Publish this manifest before accessing either holdout:",
        token(data),
    )


if __name__ == "__main__":
    main()
