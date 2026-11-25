"""Bind the sparse crop/receipt replay to its qualified CUDA sources and isolated full regression."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from anchor_audit import publish
from check_periodic_release import verify as periodic
from periodic_roi import PROTOCOL, need
from qualify_periodic_roi import CAMPAIGN, DEMOS, sha


def verify(full, edge, regions, reports, backend, study, survey, replay, out):
    with TemporaryDirectory() as tmp:
        rechecked = Path(tmp) / "rechecked.json"
        periodic(full, edge, regions, reports, backend, study, survey, rechecked)
        parent = json.loads(rechecked.read_bytes())
    need(
        parent["parcel_campaign_sha256"] == CAMPAIGN and parent["pytest"]["tests"] >= 638,
        "prior qualification or expanded test coverage missing",
    )
    names = [
        t
        for t in ElementTree.parse(full / "native-tests.xml").iterfind(".//testcase")
        if t.get("classname") == "tests.test_periodic_roi"
    ]
    need(
        len(names) >= 28 and all(not list(t) for t in names),
        "negative controls missing, skipped or failed",
    )
    digest = sha(replay / "qualification.json")
    need(
        sha(reports / "periodic-roi-qualification.json") == digest,
        "fresh ROI replay differs from archive",
    )
    root = full.parent
    q = json.loads((replay / "qualification.json").read_bytes())
    need(
        q["format"] == "CFDK-periodic-roi-qualification-v1"
        and q["protocol_sha256"]
        == PROTOCOL
        == sha(root / "fixtures/periodic-roi-receipt-protocol-v1.json")
        and q["parcel_campaign_sha256"] == CAMPAIGN == sha(reports / "periodic-parcels.json")
        and q["verifier_sha256"] == sha(root / "scripts/periodic_roi.py")
        and q["qualification_script_sha256"] == sha(root / "scripts/qualify_periodic_roi.py"),
        "different source, script or protocol",
    )
    prior = json.loads((reports / "periodic-parcels.json").read_bytes())
    need(set(q["cases"]) == set(prior["cases"]), "case scope")
    maps = cells = queries = refusals = 0
    for n, c in prior["cases"].items():
        r = q["cases"][n]
        need(r["status"] == c["status"], "different source outcome")
        if c["status"] == "upstream_refused":
            need(not r["rows"] and not (replay / n).exists(), "refused source has index")
            refusals += 1
            continue
        rows = {tuple(x["pair"]): x for x in r["rows"]}
        need(len(r["rows"]) == len(rows) == len(c["rows"]), "map count")
        for old in c["rows"]:
            i, j = old["pair"]
            row = rows[i, j]
            idx = replay / n / f"{i}-{j}.json"
            info = json.loads(idx.read_bytes())
            h, w = info["shape_ha"]
            need(
                row["parcel_manifest_sha256"]
                == info["parcel_manifest_sha256"]
                == old["manifest_sha256"]
                and sha(idx) == row["index_sha256"]
                and len(row["query_sha256"]) == 5,
                "fresh per-map index or query count",
            )
            maps += 1
            cells += h * w
            queries += len(row["query_sha256"])
    need(
        (maps, cells, queries, refusals)
        == (
            q["parcels"],
            q["partition_cells"],
            q["partition_and_point_queries"],
            q["upstream_refused"],
        )
        == (67, 988920, 335, 2),
        "ROI coverage",
    )
    examples = {}
    for (n, i, j), (tag, box) in DEMOS.items():
        z = f"roi-{tag}-receipt.zip"
        d = q["portable_examples"][tag]
        r = d["receipt"]
        need(
            r["box_y_x_h_w"] == box
            and r["pair"] == [i, j]
            and r["index_sha256"] == sha(replay / n / f"{i}-{j}.json")
            and sha(reports / z) == sha(replay / z) == d["zip_sha256"],
            "detached artifacts differ",
        )
        examples[tag] = d["zip_sha256"]
    publish(
        out,
        dict(
            scope="Isolated non-CoreOnly CT regression and replay of qualified CUDA source artifacts. Fresh standard-library ROI receipts, both portable examples and the complete 67-index survey reproduced byte for byte. Not a new GPU speed trial or dimensional accuracy qualification.",
            gpu=parent["gpu"],
            pytest=parent["pytest"],
            roi_tests=len(names),
            protocol_sha256=PROTOCOL,
            qualification_sha256=digest,
            parcel_campaign_sha256=CAMPAIGN,
            prior_cuda_qualification_sha256=parent["qualification_sha256"],
            verified_partition_cells=cells,
            queries=queries,
            portable_zip_sha256=examples,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in (
        "full",
        "edge",
        "regions",
        "reports",
        "backend",
        "study",
        "survey",
        "replay",
        "out",
    ):
        p.add_argument(n, type=Path)
    verify(**vars(p.parse_args()))
