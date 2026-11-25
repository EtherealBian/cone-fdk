"""Generate optional, versioned directory-review modules from pinned algorithms.

Do not replace the original periodic/slab overlays or modify a historical source.
Only route shared reads, native paths and publication through an invocation-local lease.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

from sealed_inputs import publish_new

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = {
    "scripts/edge_review.py": "125fdcc7abc7adcdb520f62bf7660f38975300d76dfcaf44c369aff334f24c7a",
    "scripts/edge_review_packed.py": "72ff84e39bb04de9f894910594378894ba9231288d8d4649925e663fc61c890d",
    "scripts/wall.py": "b0c1d8f832c21c8d886df961d455b6e476c98aa8514c30547b73ef872f7c2497",
    "scripts/wall_pack.py": "5ec5a9aecbf1c0e4b7ffe126b75d4b7cabf879d8b384ceb3c16e0be3c3b3ada6",
    "platform/spark/sealed_inputs.py": "16fc3e8b791d6aab57a204781e124b4d2eb8a1b98082db1e0392fddadffbdec2",
    "platform/spark/spark_publication.py": "a497b5387b56c602191da289b2dca36fbff9d58465a9c1e7060abca7a76dfa66",
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def one(text, a, b):
    if text.count(a) != 1:
        raise ValueError("unexpected upstream boundary: " + a)
    return text.replace(a, b)


def port(name, raw):
    s = raw.decode().replace("\r\n", "\n")
    s = one(
        s,
        "from axis_gauge import pinned_reader",
        "from edge_job_inputs import review as _leased_review",
    )
    if name == "edge_review.py":
        first = "def run(parent_job, job, levels, *, budget=16, parent_sha256=None):"
        body = "    return _leased_review(_run, parent_job, job, levels, products=BUILD, source_products=source_products, budget=budget, parent_sha256=parent_sha256)"
        s = one(
            s,
            first,
            first[:-2]
            + ", source_products=None):"
            + "\n"
            + body
            + "\n\n\ndef _run(parent_job, job, levels, *, products, source_products, pinned_reader, budget, parent_sha256):",
        )
        s = one(s, "BUILD / name, sources[name]", "source_products / name, sources[name]")
        s = one(s, "dest, lvl, budget=budget", "dest, lvl, budget=budget, build=products")
    else:
        s = one(
            s,
            "def prepare(parent, bind, levels, expected):",
            "def prepare(parent, bind, levels, expected, *, products=BUILD):",
        )
        s = one(s, "BUILD / name, src[name]", "products / name, src[name]")
        first = "def run(parent_job, job, levels, *, parent_sha256, companion_sha256, budget=16):"
        body = "    return _leased_review(_run, parent_job, job, levels, products=BUILD, source_products=source_products, budget=budget, parent_sha256=parent_sha256, companion_sha256=companion_sha256, packed=True)"
        s = one(
            s,
            first,
            first[:-2]
            + ", source_products=None):"
            + "\n"
            + body
            + "\n\n\ndef _run(parent_job, job, levels, *, products, source_products, pinned_reader, parent_sha256, companion_sha256, budget):",
        )
        s = one(
            s,
            "parent, bind, levels, parent_sha256\n",
            "parent, bind, levels, parent_sha256, products=source_products\n",
        )
        s = one(
            s,
            'BUILD / "wall-pack.exe", companion_sha256',
            'products / "wall-pack.exe", companion_sha256',
        )
        s = one(s, "extra, budget=budget)", "extra, budget=budget, build=products)")
    s = one(
        s,
        '    p.add_argument("parent_job", type=Path)',
        '    p.add_argument("--source-products", type=Path, help="original producer files, read only; requires external parent SHA-256")\n    p.add_argument("parent_job", type=Path)',
    )
    compile(s, name, "exec")
    return s.encode()


def build(out, expected=None):
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise ValueError("qualified for ARM64 Linux only")
    out = Path(out).absolute()
    if out.parent != out.parent.resolve(strict=True) or out.name in ("", ".."):
        raise ValueError("real build parent required")
    products = {}
    for rel, h in PINS.items():
        p = ROOT / rel
        raw = p.read_bytes()
        if sha(raw) != h:
            raise ValueError("frozen source changed: " + rel)
        products[p.name] = port(p.name, raw) if p.name.startswith("edge_review") else raw
    api = HERE / "edge_job_inputs.py"
    products[api.name] = api.read_bytes()
    manifest = {
        "format": "CFDK-GB10-directory-edge-overlay-v1",
        "sources": PINS,
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "products_sha256": {k: sha(v) for k, v in products.items()},
    }
    products["overlay.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    pin = sha(products["overlay.json"])
    if expected is not None and expected != pin:
        raise ValueError("unexpected overlay digest")
    if os.path.lexists(out):
        if (
            out.is_symlink()
            or not out.is_dir()
            or set(products) != {p.name for p in out.iterdir()}
            or any(
                (out / k).is_symlink() or not (out / k).is_file() or (out / k).read_bytes() != b
                for k, b in products.items()
            )
        ):
            raise ValueError("existing overlay not identical")
    else:
        with tempfile.TemporaryDirectory(dir=out.parent) as td:
            stage = Path(td) / "ready"
            stage.mkdir()
            for k, b in products.items():
                (stage / k).write_bytes(b)
            publish_new(stage, out)
    return pin


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--expect")
    a = p.parse_args()
    print(build(a.out, a.expect))
