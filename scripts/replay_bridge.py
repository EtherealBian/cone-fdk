"""Rebuild the preregistered bridge study and its portable batch; compare frozen bytes."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from bridge_design import SPEC_SHA256, protocol
from bridge_phantom import generate
from compare_native import sha
from region_batch import run as batch
from region_console import export
from remount_loss import write_doc
from validate_bridge import qualify
from validate_roi import require


def bundle(work, dest, expected):
    require(
        (work / "result.json").read_bytes() == expected,
        "measured study differs from the frozen result",
    )
    d = json.loads(expected)
    require(d["protocol_sha256"] == SPEC_SHA256, "wrong study protocol")
    dest.mkdir(exist_ok=False, parents=True)
    # Paths in a batch recipe are relative to it and must not escape its folder.
    recipe = work / "recipe.json"
    require(not recipe.exists(), "use fresh recipe")
    write_doc(
        recipe,
        dict(
            format="CFDK-region-batch-plan-v1",
            entries=[
                dict(
                    id=n,
                    kind="edge",
                    job=n + "/edge",
                    manifest_sha256=c["edge_sha256"],
                    area_radius_mm=protocol()["measurement"]["area_radius_mm"],
                )
                for n, c in d["cases"].items()
            ],
        ),
    )
    index, first = batch(recipe, dest / "batch")
    again, second = batch(recipe, dest / "batch", resume=True)
    require(
        index == again
        and first["computed"] == 4
        and second["reused"] == 4
        and all(
            x["result_sha256"] == d["cases"][x["id"]]["region_sha256"] for x in index["entries"]
        ),
        "batch vs individual or resume",
    )
    ui = export(recipe, dest / "batch", dest / "console", sha(dest / "batch/manifest.json"))
    require(
        ui["entries"] == 4 and ui["maps"] == 36 and ui["pair_cells"] == 531360,
        "incomplete portable result",
    )
    result = dict(
        protocol_sha256=SPEC_SHA256,
        measured_result_sha256=sha(work / "result.json"),
        console=ui,
        scientific_gates=d["acceptance"],
    )
    write_doc(dest / "bundle.json", result)
    return result


def replay(base, expected, expected_bundle, evidence):
    require(not evidence.exists() and not base.exists(), "require fresh run and evidence paths")
    source = expected.read_bytes()
    portable = expected_bundle.read_bytes()
    base.mkdir(parents=True)
    generate(base / "scans")
    qualify(base / "scans", base / "measurement", json.loads(source)["companion_sha256"])
    b = bundle(base / "measurement", base / "portable", source)
    require(
        (base / "portable/bundle.json").read_bytes() == portable,
        "offline result differs from archived batch/console",
    )
    write_doc(
        evidence,
        dict(
            protocol="Fresh analytic projections through four witnessed comparisons, packed global thresholds, independent native and CPU labels, resumable batch and offline console; compared to prior bytes, not physical measurements.",
            **b,
        ),
    )
    print(
        "PASS: independent detector-to-bundle replay; scientific gates:",
        b["scientific_gates"]["passed"],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("expected", "expected_bundle", "evidence"):
        p.add_argument(k, type=Path)
    p.add_argument("--keep", type=Path)
    a = p.parse_args()
    with TemporaryDirectory() as tmp:
        replay(a.keep or Path(tmp) / "run", a.expected, a.expected_bundle, a.evidence)
