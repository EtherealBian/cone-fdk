"""Verify fresh detector-to-gauge replay plus unchanged frozen outputs on the qualified build."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from axis_figure import draw
from check_loss_regression import verify as verify_loss
from common import REPORT
from compare_native import sha
from validate_roi import require
from validate_stack import save_json


def verify(full, replay, baseline):
    old = verify_loss(full, replay, baseline)
    require(old["pytest"]["tests"] >= 215, "axis regression tests missing")

    def load(p):
        return json.loads(p.read_bytes())

    loss = load(full / "wall-loss.json")
    frozen = load(baseline / "wall-loss.json")
    loss.pop("binary_sha256")
    frozen.pop("binary_sha256")
    require(loss == frozen, "published loss metrics or payload changed")
    for n in (
        "wall-loss-example.json",
        "wall-loss-scale.json",
        "wall-loss-demo.zip",
        "wall-loss.png",
    ):
        require(sha(full / n) == sha(baseline / n), "frozen evidence changed: " + n)
    build = old["build"]
    q = load(full / "axis-gauge.json")
    pack = full / "axis-review.zip"
    require(
        q["binary_sha256"]
        == {"cone-fdk.exe": build["executable_sha256"], "kernels.ptx": build["ptx_sha256"]}
        and sha(pack) == q["review_sha256"],
        "axis qualification build or pack mismatch",
    )
    with zipfile.ZipFile(pack) as z:
        job = json.loads(z.read("manifest.json"))
        case = q["cases"]["tilted-second-part"]
        require(
            job["status"] == "measured"
            and job["inputs_sha256"][".f32"] == case["projection_sha256"]
            and job["outputs_sha256"] == case["artifacts_sha256"],
            "wrong example job",
        )
        for n in z.namelist():
            if n != "manifest.json":
                require(
                    hashlib.sha256(z.read(n)).hexdigest() == job["outputs_sha256"][n],
                    "bad pack member: " + n,
                )
    for root in (full, replay):
        draw(root / "axis-review.zip", root / "axis-profiles.png")
    hashes = {
        n: sha(full / n) for n in ("axis-gauge.json", "axis-review.zip", "axis-profiles.png")
    }
    require(
        all(sha(replay / n) == v for n, v in hashes.items()), "independent axis replay mismatch"
    )
    return dict(
        protocol="Full acceptance, independent simulation replay (axis, wall, bore, loss) and frozen legacy output comparison. No independent physical scan.",
        gpu=old["gpu"],
        pytest=old["pytest"],
        build=build,
        axis_replay_sha256=hashes,
        frozen_loss_reproduced=True,
        legacy=old,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("full", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("--baseline", default=REPORT, type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    save_json(a.out, verify(a.full, a.replay, a.baseline))
    print("PASS: detector-to-gauge replay and legacy outputs")
