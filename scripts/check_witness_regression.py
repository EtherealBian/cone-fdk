"""Verify a fresh full run, detector replay, offline centroid pack, and frozen pre-guard results."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from check_remount_regression import verify as verify_native
from common import REPORT
from compare_native import sha
from remount_loss import write_doc
from validate_roi import require
from witness_figure import draw
from witness_probe import audit


def load(path):
    return json.loads(path.read_bytes())


def verify(full, replay, remount_replay, historical):
    native = verify_native(full, remount_replay, historical / "axis-regression.json")
    require(native["pytest"]["tests"] >= 270, "missing witness tests")
    # Keep the previous campaign as published; compare computational payloads separately from provenance.
    prior = load(historical / "remount-regression.json")["remount_replay_sha256"]
    require(
        all(sha(historical / n) == v for n, v in prior.items()), "historical campaign mismatch"
    )
    for n in ("remount-deepened.zip", "remount-loss.png"):
        require(sha(full / n) == prior[n], "historical sampled wall evidence changed: " + n)
    old, new = (load(root / "remount-example-job.json") for root in (historical, full))
    old_locator, new_locator = [
        j["locator_software"]["source_sha256"].pop("remount_loss.py") for j in (old, new)
    ]
    require(old == new, "old example job changed beyond locator source fingerprint")
    old, new = (load(root / "remount-loss.json") for root in (historical, full))
    for doc in (old, new):
        doc["example_sha256"].pop("remount-example-job.json")
        for case in doc["cases"].values():
            case.pop("manifest_sha256")
    require(old == new, "legacy remount metrics, input or loss payload changed")
    for root in (full, replay):
        pack = root / "witness-pivot.zip"
        report = load(root / "body-witness.json")
        require(sha(pack) == report["example_pivot_zip_sha256"], "unbound witness evidence pack")
        with ZipFile(pack) as z:
            require(
                hashlib.sha256(z.read("manifest.json")).hexdigest()
                == report["cases"]["pivot"]["manifest_sha256"],
                "wrong pivot manifest",
            )
        require(
            [c["status"] for c in audit(pack)] == ["pass", "displaced", "displaced"],
            "CPU centroid replay",
        )
        draw(pack, root / "body-witness.png")
    hashes = {
        n: sha(full / n)
        for n in (
            "body-witness.json",
            "witness-margin.json",
            "witness-pivot.zip",
            "body-witness.png",
        )
    }
    require(all(sha(replay / n) == v for n, v in hashes.items()), "witness replay mismatch")
    return dict(
        protocol="Clean repeat native build and full acceptance; second freshly generated witness projection set. Prior fixture-only campaign kept frozen.",
        pytest=native["pytest"],
        gpu=native["gpu"],
        build=native["build"],
        witness_replay_sha256=hashes,
        historical_fixture_only_results_equal_except_provenance=True,
        historical_locator_sha256=old_locator,
        current_locator_sha256=new_locator,
        historical_remount_regression_sha256=sha(historical / "remount-regression.json"),
        current_remount_replay_sha256=native["remount_replay_sha256"],
        legacy_wall_bore_axis_and_loss_verified=True,
        frozen_volumes_sha256=native["frozen_volumes_sha256"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("full", "replay", "remount_replay"):
        p.add_argument(n, type=Path)
    p.add_argument("--historical", type=Path, default=REPORT)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    write_doc(a.out, verify(a.full, a.replay, a.remount_replay, a.historical))
    print(
        "PASS: independent witness replay, previous fixture-only payloads and native regression"
    )
