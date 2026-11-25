"""Bind the preregistered bridge replay to the non-CoreOnly and prior packed regression gates."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from bridge_design import SPEC_SHA256, predict, protocol
from check_packed_regression import verify as packed
from check_region_regression import verify as regions
from compare_native import sha
from remount_loss import write_doc
from validate_bridge import assess
from validate_roi import require


def verify(full, edge_replay, region_replay, reports):
    with TemporaryDirectory() as tmp:
        rr = Path(tmp) / "regions.json"
        write_doc(rr, regions(full, edge_replay, region_replay, reports))
        old = packed(
            full,
            reports / "packed-edge-qualification.json",
            reports / "packed-edge-ab.json",
            rr,
            reports / "specimen-replay.json",
        )
    found = list(full.glob("bridge-replay-*.json"))
    require(len(found) == 1, "one fresh bridge replay required")
    record = json.loads(found[0].read_bytes())
    record.pop("protocol")
    archived = json.loads((reports / "bridge-bundle.json").read_bytes())
    measured = json.loads((reports / "bridge-result.json").read_bytes())
    require(
        record == archived
        and record["protocol_sha256"] == measured["protocol_sha256"] == SPEC_SHA256
        and predict() == json.loads((reports / "bridge-design.json").read_bytes()),
        "protocol or replay differs",
    )
    require(
        record["scientific_gates"] == assess(measured["cases"], protocol())
        and record["measured_result_sha256"] == sha(reports / "bridge-result.json")
        and record["console"]["html_sha256"] == sha(reports / "bridge-console.html"),
        "result, scientific gate or offline digest differs",
    )
    require(old["full_pipeline_pytest"]["tests"] >= 444, "missing bridge tests")
    return dict(
        protocol="Clean index checkout, two native builds and full non-CoreOnly acceptance with Packed, Specimen and Bridge campaigns enabled. Prior original and packed evidence and real walnut volume digests checked via historical regression gates. Frozen simulated bridge design, detector-to-console replay and returned scientific outcomes checked independently.",
        gpu=old["gpu"],
        pytest=old["full_pipeline_pytest"],
        prior_region_regression_sha256=old["full_region_regression_sha256"],
        old_witness_bundle_sha256=old["fresh_and_archived_packed_qualification_sha256"],
        old_specimen_replay_sha256=old["fresh_and_archived_specimen_replay_sha256"],
        bridge_replay_sha256=sha(found[0]),
        **record,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("full", "edge_replay", "region_replay", "reports", "out"):
        p.add_argument(n, type=Path)
    args = p.parse_args()
    write_doc(args.out, verify(args.full, args.edge_replay, args.region_replay, args.reports))
    print(
        "PASS: original regression, benchmark recheck, three campaigns and detector-to-console replay"
    )
