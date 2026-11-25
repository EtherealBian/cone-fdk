"""Replay a five-entry batch against independent single-job and witness qualification records."""

import argparse
import json
from pathlib import Path

from compare_native import sha
from region_batch import run
from remount_loss import write_doc
from validate_roi import require


def qualify(recipe, output, trusted, dest):
    first, fresh = run(recipe, output)
    index = sha(output / "manifest.json")
    second, resumed = run(recipe, output, resume=True)
    require(
        first == second
        and index == sha(output / "manifest.json")
        and fresh == dict(computed=5, reused=0, input_errors=0)
        and resumed == dict(computed=0, reused=5, input_errors=0),
        "resume changed results or recomputed a job",
    )
    require(
        first["counts"]
        == dict(complete_coverage=2, incomplete_coverage=1, upstream_refused=2, input_error=0),
        "wrong batch dispositions",
    )
    regions = json.loads((trusted / "region-qualification.json").read_bytes())["cases"]
    witness = json.loads((trusted / "body-witness.json").read_bytes())["cases"]
    rows = {}
    for r in first["entries"]:
        name = r["id"]
        path = output / r["result_file"]
        require(sha(path) == r["result_sha256"], "content address mismatch")
        if name in regions:
            require(
                r["source_manifest_sha256"] == regions[name]["edge_sha256"]
                and r["result_sha256"] == regions[name]["region_sha256"],
                "not the independent single-job result",
            )
        else:
            w = witness[name]
            obj = json.loads(path.read_bytes())
            require(
                obj["witness_manifest_sha256"] == w["manifest_sha256"]
                and obj["reason"] == w["reason"]
                and obj["rejected_scan"] == w["rejected_scan"]
                and r["status"] == "upstream_refused",
                "wrong upstream refusal",
            )
        rows[name] = {
            k: r[k] for k in ("kind", "status", "source_manifest_sha256", "result_sha256")
        }
    write_doc(
        dest,
        dict(
            protocol="Five simulated inspection jobs, not five independent specimens. Fresh batch plus fully revalidated resume; all three region documents match independently qualified single jobs.",
            recipe_sha256=first["recipe_sha256"],
            engine_sha256=first["engine_sha256"],
            manifest_sha256=index,
            counts=first["counts"],
            fresh=fresh,
            resume=resumed,
            entries=rows,
        ),
    )
    print("PASS: 5 inspected entries / 5 revalidated cache hits / identical batch manifest")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("recipe", "output", "trusted", "dest"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
