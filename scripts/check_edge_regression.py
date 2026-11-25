"""Bind a clean full acceptance, archived-parent replay and the published raw wall pack."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from common import REPORT
from compare_native import sha
from remount_loss import write_doc
from validate_roi import require


def load(path):
    return json.loads(path.read_bytes())


def verify(full, replay, baseline):
    frozen = load(baseline / "witness-regression.json")
    tests = ET.parse(full / "native-tests.xml").find("testsuite").attrib
    tests = {k: int(tests[k]) for k in ("tests", "errors", "skipped", "failures")}
    require(
        tests["tests"] >= 304
        and not any(tests[k] for k in ("errors", "skipped", "failures"))
        and (full / "RESULTS.md").is_file(),
        "incomplete acceptance",
    )
    build = load(full / "build-repro.json")
    require(build == frozen["build"] and build["identical"], "build identity")
    volumes = {
        f"analytic{k}": v["sha256"]["native"] for k, v in load(full / "phantom.json").items()
    }
    walnut = load(full / "walnut-full.json")
    stress = load(full / "oversize.json")
    volumes.update(walnut512=walnut["output_sha256"], stress1296=stress["sha256"])
    require(volumes == frozen["frozen_volumes_sha256"], "volume regression")
    # Historical PNG for the three witness windows is produced by a separate illustration step.
    prior = {
        **frozen["current_remount_replay_sha256"],
        **{k: v for k, v in frozen["witness_replay_sha256"].items() if k != "body-witness.png"},
    }
    require(all(sha(full / n) == v for n, v in prior.items()), "parent qualification changed")
    names = [
        "edge-cell.json",
        "edge-review-example.zip",
        "edge-review.png",
        "edge-qualification.json",
    ]
    hashes = {n: sha(full / n) for n in names}
    require(
        all(sha(replay / n) == v for n, v in hashes.items()), "archived/fresh replay mismatch"
    )
    evidence = load(full / "edge-qualification.json")
    w = load(full / "body-witness.json")["cases"]
    require(evidence["evidence_sha256"] == {n: hashes[n] for n in names[:-1]}, "unbound evidence")
    for case, src in [
        ("noisy-change", "noisy-change"),
        ("repeat", "repeat"),
        ("invalid-edge-control", "repeat"),
    ]:
        require(
            evidence["cases"][case]["parent_manifest_sha256"] == w[src]["manifest_sha256"],
            "wrong parent manifest",
        )
    with zipfile.ZipFile(full / "edge-review-example.zip") as z:
        raw = z.read("manifest.json")
        doc = json.loads(raw)
        gpu = {
            s["gpu"]
            for stages in doc["stages"].values()
            for s in stages
            if s["source"] == "cuda_wall"
        }
        require(
            hashlib.sha256(raw).hexdigest()
            == evidence["cases"]["noisy-change"]["review_manifest_sha256"]
            and doc["inputs_sha256"]["manifest.json"] == w["noisy-change"]["manifest_sha256"]
            and doc["inputs_sha256"]["cone-fdk.exe"] == build["executable_sha256"]
            and doc["inputs_sha256"]["kernels.ptx"] == build["ptx_sha256"]
            and gpu == {frozen["gpu"]},
            "unbound GPU review",
        )
        require(
            set(z.namelist()) == {"manifest.json", *doc["outputs_sha256"]}
            and all(
                hashlib.sha256(z.read(n)).hexdigest() == v
                for n, v in doc["outputs_sha256"].items()
            ),
            "bad offline map",
        )
    return dict(
        protocol="All acceptance stages on a clean repeat native build. Revised pytest and integrated witness/edge stage rerun on that build; edge outputs compared byte-for-byte with a replay using archived witness jobs.",
        gpu=gpu.pop(),
        pytest=tests,
        build=build,
        frozen_volumes_sha256=volumes,
        prior_remount_and_witness_sha256=prior,
        edge_replay_sha256=hashes,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("full", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("--baseline", type=Path, default=REPORT)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    write_doc(a.out, verify(a.full, a.replay, a.baseline))
    print("PASS: edge evidence replay, parent campaigns, native build and volumes")
