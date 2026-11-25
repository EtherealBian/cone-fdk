"""Replay ten striped dispositions, a real process interruption, and an independent dense audit."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from anchor_audit import audit, publish
from compare_native import sha
from stripe_batch import run
from validate_roi import require


def read(p):
    return json.loads(p.read_bytes())


def qualify(recipe, reports, work, evidence):
    require(not work.exists() and not evidence.exists(), "fresh work and evidence required")
    work.mkdir(parents=True)
    cold, first = run(recipe, work / "cold")
    index = sha(work / "cold/manifest.json")
    hot, resumed = run(recipe, work / "cold", resume=True)
    require(
        hot == cold
        and sha(work / "cold/manifest.json") == index
        and first == dict(computed=10, reused=0, revalidated_payload_bytes=0, input_errors=0)
        and resumed
        == dict(computed=0, reused=10, revalidated_payload_bytes=23734080, input_errors=0),
        "bad cold/resume accounting",
    )
    require(
        cold["counts"]
        == dict(complete_coverage=7, incomplete_coverage=1, upstream_refused=2, input_error=0),
        "wrong case dispositions",
    )
    # Only the qualification child injects failure; there is no production CLI to fake a crash.
    driver = """import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import stripe_batch as b
original=b.review
calls=0
def stop(*args):
    global calls
    calls+=1
    if calls==5: os._exit(73)
    return original(*args)
b.review=stop
b.run(Path(sys.argv[2]),Path(sys.argv[3]))"""
    interrupted = work / "interrupted"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            driver,
            str(Path(__file__).resolve().parent),
            str(recipe),
            str(interrupted),
        ],
        capture_output=True,
        timeout=120,
    )
    require(
        child.returncode == 73 and not (interrupted / "manifest.json").exists(),
        "child did not stop between committed cases",
    )
    recovered, partial = run(recipe, interrupted, resume=True)
    require(
        recovered == cold
        and sha(interrupted / "manifest.json") == index
        and partial
        == dict(computed=6, reused=4, revalidated_payload_bytes=12339360, input_errors=0),
        "interrupted batch differs",
    )
    specimens = read(reports / "specimen-qualification.json")
    stripe = read(reports / "stripe-qualification.json")["cases"]
    witness = read(reports / "body-witness.json")["cases"]
    recipe_rows = {e["id"]: e for e in read(recipe)["entries"]}
    rows = {}
    maps = cells = 0
    for row in cold["entries"]:
        n = row["id"]
        e = recipe_rows[n]
        p = work / "cold" / row["result_file"]
        obj = read(p)
        require(
            sha(p) == sha(interrupted / row["result_file"]) == row["result_sha256"]
            and row["source_manifest_sha256"] == e["manifest_sha256"],
            "different recovered artifact or source",
        )
        if n in stripe:
            require(
                obj["result"] == stripe[n]["result"],
                "not the independently qualified native topology",
            )
        elif n in specimens["cases"]:
            c = specimens["cases"][n]
            r = obj["result"]
            a = audit(
                reports / "specimen-console.html",
                specimens["console"]["html_sha256"],
                n,
                r["anchors_row_col"],
            )
            require(
                c["edge_manifest_sha256"] == e["manifest_sha256"] == a["edge_sha256"]
                and c["region_manifest_sha256"] == a["region_sha256"],
                "independent specimen lineage",
            )
            for x, s, t in zip(r["scenarios"], c["rows"], a["scenarios"], strict=True):
                sizes = [v["measured_cells"] for v in s["components"]]
                require(
                    x["pair"] == s["pair"] == t["pair"]
                    and x["anchor_state"] == t["state"]
                    and x["observed_components"] == len(sizes)
                    and x["largest_observed_cells"] == max(sizes, default=0)
                    and x["observed_flagged_cells"] == sum(sizes)
                    and x["unknown_cells"] == 0,
                    "independent specimen measurements differ",
                )
        else:
            require(
                n in ("slide", "missing-reference")
                and obj["witness_manifest_sha256"] == witness[n]["manifest_sha256"]
                and obj["reason"] == witness[n]["reason"]
                and obj["rejected_scan"] == witness[n]["rejected_scan"]
                and "anchor_states" not in row,
                "refusal incorrectly converted",
            )
        if e["kind"] == "edge":
            h, w = obj["result"]["shape_ha"]
            maps += row["pairs"]
            cells += h * w * row["pairs"]
        rows[n] = row
    require((maps, cells) == (67, 988920), "unexpected scope")
    publish(
        evidence,
        dict(
            scope="Ten jobs are not ten independent specimens. Simulated projection-derived comparisons and two upstream refusals. Real OS process exit after four committed jobs; full and interrupted redo return the same index bytes.",
            recipe_sha256=cold["recipe_sha256"],
            engine_sha256=cold["engine_sha256"],
            validator_sha256=sha(Path(__file__)),
            source_records_sha256={
                n: sha(reports / n)
                for n in (
                    "stripe-qualification.json",
                    "specimen-qualification.json",
                    "body-witness.json",
                )
            },
            maps=maps,
            pair_cells=cells,
            counts=cold["counts"],
            cold=first,
            resume=resumed,
            after_four_commits=partial,
            manifest_sha256=index,
            entries=rows,
        ),
    )
    print(
        "PASS: 10 jobs, 67 paired grids, 4 committed before injected exit; 6 recomputed and full index restored"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("recipe", "reports", "work", "evidence"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
