"""Verify and preserve the mixed-batch release receipt from a real clean-checkout replay."""

import argparse
import json
from pathlib import Path

from anchor_audit import publish
from check_bridge_regression import verify as bridge
from compare_native import sha
from validate_roi import require


def read(p):
    return json.loads(p.read_bytes())


def verify(full, edge, regions, reports, work, replay, out):
    old = bridge(full, edge, regions, reports)
    require(
        old["pytest"]["tests"] >= 538
        and all(not old["pytest"][k] for k in ("skipped", "errors", "failures")),
        "missing batch full tests",
    )
    q = read(reports / "stripe-batch-qualification.json")
    r = read(replay)
    require(
        r["verifier_sha256"] == sha(full.parent / "scripts/replay_stripe_mixed.py")
        and q["validator_sha256"] == sha(full.parent / "scripts/qualify_stripe_batch.py")
        and sha(work / "fixtures/stripe-campaign-plan.json")
        == q["recipe_sha256"]
        == r["recipe_sha256"],
        "unbound recipe or replay scripts",
    )
    matched = r["matched_sha256"]
    require(
        set(matched)
        == {
            "body-witness.json",
            "edge-qualification.json",
            "region-qualification.json",
            "packed-edge-qualification.json",
            "specimen-qualification.json",
            "specimen-console.html",
            "bridge-result.json",
            "bridge-bundle.json",
            "bridge-console.html",
            "anchor-qualification.json",
            "stripe-qualification.json",
            "stripe-batch-qualification.json",
        }
        and all(sha(reports / n) == v for n, v in matched.items()),
        "missing or different comparison",
    )
    require(
        sha(work / "new-reports/stripe-mixed.json") == matched["stripe-batch-qualification.json"]
        and q["manifest_sha256"] == r["final_manifest_sha256"]
        and (q["maps"], q["pair_cells"]) == (r["maps"], r["pair_cells"]) == (67, 988920),
        "mixed scope or provenance mismatch",
    )
    for method in ("cold", "interrupted"):
        folder = work / "journal-test" / method
        manifest = folder / "manifest.json"
        require(sha(manifest) == q["manifest_sha256"], "batch manifest differs")
        doc = read(manifest)
        require(
            doc["counts"] == q["counts"]
            and len(doc["entries"]) == 10
            and {row["id"]: row for row in doc["entries"]} == q["entries"],
            "batch dispositions differ",
        )
        for row in doc["entries"]:
            digest = row["result_sha256"]
            require(
                row["result_file"] == "results/" + digest + ".json"
                and sha(folder / "results" / (digest + ".json")) == digest,
                "individual result differs",
            )
    publish(
        out,
        dict(
            scope="Two identical native builds, full non-CoreOnly regression with packed, specimen and bridge campaigns. Independent clean simulation-to-batch replay. Verified the ten committed result payloads in both uninterrupted and OS-interrupted journals.",
            gpu=old["gpu"],
            pytest=old["pytest"],
            retained_historical_bridge_sha256=old["bridge_replay_sha256"],
            prior_region_regression_sha256=old["prior_region_regression_sha256"],
            batch_validation_sha256=matched["stripe-batch-qualification.json"],
            full_mixed_replay_sha256=sha(replay),
            detector_to_batch_matched_files=len(matched),
            restored_index_sha256=q["manifest_sha256"],
            source_maps=q["maps"],
            pair_cells=q["pair_cells"],
        ),
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    for n in ("full", "edge", "regions", "reports", "work", "replay", "out"):
        a.add_argument(n, type=Path)
    verify(**vars(a.parse_args()))
