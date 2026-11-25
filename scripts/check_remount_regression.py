"""Bind an independent remount replay to a full acceptance run and the frozen axis baseline."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from axis_figure import draw
from check_loss_regression import sections
from common import REPORT
from compare_native import sha
from remount_loss import write_doc
from validate_roi import require


def verify(full, replay, baseline):
    def load(name):
        return json.loads((full / (name + ".json")).read_bytes())

    old = json.loads(baseline.read_bytes())
    prior = old["legacy"]
    build = load("build-repro")
    m = load("remount-loss")
    job = load("remount-example-job")
    junit = ET.parse(full / "native-tests.xml").find("testsuite").attrib
    tests = {k: int(junit[k]) for k in ("tests", "errors", "failures", "skipped")}
    require(
        (full / "RESULTS.md").is_file()
        and tests["tests"] >= 239
        and not any(tests[k] for k in ("errors", "failures", "skipped")),
        "full acceptance missing",
    )
    require(
        build == old["build"]
        and build["identical"]
        and m["binary_sha256"]
        == {"cone-fdk.exe": build["executable_sha256"], "kernels.ptx": build["ptx_sha256"]},
        "native binary mismatch",
    )
    hashes = {n: sha(full / n) for n in ["remount-loss.json", *m["example_sha256"]]}
    require(
        all(sha(replay / n) == v for n, v in hashes.items())
        and all(hashes[n] == v for n, v in m["example_sha256"].items()),
        "remount replay mismatch",
    )
    require(
        job["status"] == "compared"
        and m["cases"]["deepened"]["manifest_sha256"] == hashes["remount-example-job.json"],
        "wrong example manifest",
    )
    with zipfile.ZipFile(full / "remount-deepened.zip") as z:
        for member, source in [
            ("grid.txt", "material.grid"),
            ("reference.wall", "reference/material.wall"),
            ("inspected.wall", "inspected/material.wall"),
            ("regions.json", "material-loss.json"),
        ]:
            require(
                hashlib.sha256(z.read(member)).hexdigest() == job["outputs_sha256"][source],
                "wrong offline review member",
            )
    draw(full / "axis-review.zip", full / "axis-profiles.png")
    frozen = {**old["axis_replay_sha256"], **prior["loss_replay_sha256"]}
    require(all(sha(full / n) == v for n, v in frozen.items()), "axis/loss regression")
    wall = load("radial-wall")
    wall_hashes = {k: v["wall_sha256"] for k, v in wall["cases"].items()}
    wall_hashes["lined-tube"] = wall["negative_control"]["wall_sha256"]
    require(
        wall_hashes == prior["legacy_wall"] and sections(full) == prior["frozen_bore_sections"],
        "wall/bore regression",
    )
    volumes = {f"analytic{k}": v["sha256"]["native"] for k, v in load("phantom").items()}
    volumes.update(
        walnut512=load("walnut-full")["output_sha256"], stress1296=load("oversize")["sha256"]
    )
    require(volumes == prior["legacy"]["legacy_full"], "volume regression")
    return dict(
        protocol="Full acceptance on a duplicate clean build; independent fresh detector-to-remount replay; selected frozen payloads compared to axis-regression.json.",
        pytest=tests,
        gpu=old["gpu"],
        build=build,
        previous_baseline_sha256=sha(baseline),
        frozen_volumes_sha256=volumes,
        frozen_axis_and_loss_sha256=frozen,
        frozen_wall_and_bore_payloads_equal=True,
        remount_replay_sha256=hashes,
        review_pack_bound_to_manifest=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("full", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("--baseline", default=REPORT / "axis-regression.json", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    write_doc(a.out, verify(a.full, a.replay, a.baseline))
    print("PASS: remount replay and native regression")
