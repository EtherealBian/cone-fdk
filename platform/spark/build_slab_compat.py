"""Build or verify a separate pinned Spark slab API overlay; never replace a prior build."""

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
    "scripts/slab_replay.py": "0fbe3430ff434c5c198befac7018dcaed6ea4b4a6d4ce903db2ecce25505ce31",
    "platform/spark/replay_spark.py": "8a71fc836318c6e9cc5c0916bc616d19dd25857fb1fb7230829ba79e67f3fd52",
    "platform/spark/sealed_inputs.py": "16fc3e8b791d6aab57a204781e124b4d2eb8a1b98082db1e0392fddadffbdec2",
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def checked(root, relative):
    raw = (root / relative).read_bytes()
    if sha(raw) != PINS[relative]:
        raise ValueError("frozen source changed: " + relative)
    return raw


def build(out, expected=None):
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise ValueError("qualified for ARM64 Linux only")
    out = Path(out).absolute()
    if out.parent != out.parent.resolve(strict=True) or out.name in ("", ".."):
        raise ValueError("real build parent required")
    protocol, worker, seals = (checked(ROOT, name) for name in PINS)
    # Only replace the import routing; the qualified worker body is unmodified.
    old = b"sys.path.insert(0, str(ROOT / 'scripts'))\nimport slab_replay as sr\nfrom sealed_inputs import publish_new, snapshot"
    if worker.count(old) != 1:
        raise ValueError("worker import boundary")
    core = worker.replace(
        old,
        b"import _spark_slab_protocol as sr\nfrom _spark_slab_seals import publish_new, snapshot",
    )
    cli = b'if __name__ == "__main__":'
    assert protocol.count(cli) == 1
    api = (HERE / "slab_compat_api.py").read_bytes()
    # Keep all original public functions, including the portable verifier and CLI.
    facade = (
        b"from _spark_slab_protocol import *\nfrom _spark_slab_api import generate\n\n"
        + cli
        + protocol.split(cli)[1]
    )
    products = {
        "_spark_slab_protocol.py": protocol,
        "_spark_slab_core.py": core,
        "_spark_slab_seals.py": seals,
        "_spark_slab_api.py": api,
        "slab_replay.py": facade,
    }
    for k, b in products.items():
        compile(b, k, "exec")
    manifest = {
        "format": "CFDK-GB10-slab-api-overlay-v2",
        "source_sha256": PINS
        | {
            "platform/spark/slab_compat_api.py": sha(api),
            "platform/spark/build_slab_compat.py": sha(Path(__file__).read_bytes()),
        },
        "products_sha256": {k: sha(v) for k, v in products.items()},
    }
    products["overlay.json"] = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    pin = sha(products["overlay.json"])
    if expected is not None and expected != pin:
        raise ValueError("unexpected overlay SHA-256")
    if os.path.lexists(out):
        if (
            out.is_symlink()
            or not out.is_dir()
            or {p.name for p in out.iterdir()} != set(products)
            or any(
                (out / k).is_symlink() or not (out / k).is_file() or (out / k).read_bytes() != b
                for k, b in products.items()
            )
        ):
            raise ValueError("existing overlay not identical")
    else:
        with tempfile.TemporaryDirectory(dir=out.parent) as temp:
            stage = Path(temp) / "stage"
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
