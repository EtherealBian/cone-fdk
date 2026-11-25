"""Reproduce the mixed striped campaign from synthetic detector rays, not retained local job paths."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from anchor_audit import publish
from bridge_phantom import generate as bridge_scans
from compare_native import sha
from qualify_anchors import qualify as check_anchors
from qualify_stripe_batch import qualify as check_batch
from qualify_striped_jobs import qualify as check_jobs
from replay_bridge import bundle
from specimen_phantom import generate as specimen_scans
from validate_bridge import qualify as measure_bridge
from validate_edges import qualify as edges
from validate_packed_edges import qualify as packed
from validate_regions import qualify as regions
from validate_roi import require
from validate_specimens import qualify as measure_specimens
from validate_witness import qualify as witnesses


def layout(base, reports, recipe, output):
    base, reports, recipe, output = (Path(p).resolve() for p in (base, reports, recipe, output))
    require(
        not base.exists() and not output.exists() and output.parent.is_dir(),
        "require new work directory and evidence file",
    )
    require(
        not base.is_relative_to(reports)
        and not reports.is_relative_to(base)
        and not recipe.is_relative_to(base)
        and not output.is_relative_to(base)
        and not base.is_relative_to(output),
        "work must be disjoint from references and output",
    )
    return base, reports, recipe, output


def same(actual, trusted):
    require(
        actual.read_bytes() == trusted.read_bytes(), "different replay bytes: " + trusted.name
    )
    return sha(trusted)


def replay(base, reports, recipe, output):
    import json

    base, reports, recipe, output = layout(base, reports, recipe, output)
    frozen = json.loads((reports / "stripe-batch-qualification.json").read_bytes())
    require(sha(recipe) == frozen["recipe_sha256"], "not the qualified mixed recipe")
    base.mkdir(parents=True, exist_ok=False)
    f = base / "fixtures"
    f.mkdir()
    local = f / "stripe-campaign-plan.json"
    shutil.copyfile(recipe, local)
    require(sha(local) == frozen["recipe_sha256"], "recipe copy changed")
    work = base / "build"
    witness = work / "witness-final-replay"
    rc = work / "regions-campaign"
    sk = base / "new-reports"
    work.mkdir()
    rc.mkdir()
    sk.mkdir()
    comparisons = {}

    def match(actual, name):
        comparisons[name] = same(actual, reports / name)

    witnesses(witness, sk)
    match(sk / "body-witness.json", "body-witness.json")
    singles = work / "edge-final-replay"
    edges(witness, singles, sk)
    match(sk / "edge-qualification.json", "edge-qualification.json")
    native = rc / "r1"
    regions(singles, native, sk, reports / "edge-qualification.json")
    match(sk / "region-qualification.json", "region-qualification.json")
    low = rc / "packed-edge-r1"
    packed(witness, singles, reports, low)
    match(low / "qualification.json", "packed-edge-qualification.json")
    spec = rc / "specimen-qualified-r1"
    acq = rc / "specimen-scans-r1"
    specimen_scans(acq)
    measure_specimens(acq, spec)
    match(spec / "qualification.json", "specimen-qualification.json")
    match(spec / "console/index.html", "specimen-console.html")
    study = work / "bridge-campaign/r1"
    acq = study / "scans"
    b = study / "measurement"
    bridge_scans(acq)
    prior = (reports / "bridge-result.json").read_bytes()
    measure_bridge(acq, b, json.loads(prior)["companion_sha256"])
    match(b / "result.json", "bridge-result.json")
    bundle(b, study / "portable", prior)
    match(study / "portable/bundle.json", "bridge-bundle.json")
    match(study / "portable/console/index.html", "bridge-console.html")
    check_anchors(reports, b, native, sk / "anchors")
    match(sk / "anchors/qualification.json", "anchor-qualification.json")
    check_jobs(reports, b, low, native, sk / "stripe-jobs.json")
    match(sk / "stripe-jobs.json", "stripe-qualification.json")
    check_batch(local, reports, base / "journal-test", sk / "stripe-mixed.json")
    match(sk / "stripe-mixed.json", "stripe-batch-qualification.json")
    require(
        sha(local) == sha(recipe)
        and all(sha(reports / name) == digest for name, digest in comparisons.items()),
        "reference changed during replay",
    )
    publish(
        output,
        dict(
            protocol="Fresh simulated detector-to-mixed-batch replay, including dense and streaming comparisons and actual process interruption. No retained scan directories required; independent external reference hashes remain necessary.",
            verifier_sha256=sha(Path(__file__)),
            recipe_sha256=sha(recipe),
            matched_sha256=comparisons,
            maps=frozen["maps"],
            pair_cells=frozen["pair_cells"],
            final_manifest_sha256=frozen["manifest_sha256"],
        ),
    )
    print(
        "PASS: detector projections through all mixed batch dispositions, certificates and restored manifest",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", type=Path)
    p.add_argument("recipe", type=Path)
    p.add_argument("evidence", type=Path)
    p.add_argument("--keep", type=Path)
    a = p.parse_args()
    with TemporaryDirectory() as tmp:
        replay(a.keep or Path(tmp) / "run", a.reports, a.recipe, a.evidence)
