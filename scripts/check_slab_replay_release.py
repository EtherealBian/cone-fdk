"""Require the fresh full CT run and direct comparison of interrupted/cold deliveries to dense data."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from anchor_audit import publish
from check_interface_release import verify as interface
from qualify_slab_replay import PLAN_PIN, REFERENCE, RULES, compare_outputs, sha
from slab_replay import need, plan, size
from slab_replay import verify as verify_volume


def verify(
    full,
    edge,
    regions,
    reports,
    backend,
    study,
    parcels,
    roi,
    build,
    trial,
    campaign,
    recovery,
    out,
):
    with TemporaryDirectory() as td:
        p = Path(td) / "prior.json"
        interface(
            full, edge, regions, reports, backend, study, parcels, roi, build, trial, campaign, p
        )
        prior = json.loads(p.read_bytes())
    tests = [
        t
        for t in ElementTree.parse(full / "native-tests.xml").iterfind(".//testcase")
        if t.get("classname") == "tests.test_slab_replay"
    ]
    need(
        len(tests) >= 44 and all(not list(t) for t in tests) and prior["pytest"]["tests"] >= 707,
        "slab controls missing or skipped",
    )
    root = full.parent
    q = json.loads((recovery / "qualification.json").read_bytes())
    pp = root / "fixtures/walnut512-slab-replay-v1.json"
    d = plan(pp, PLAN_PIN)
    need(
        q["format"] == "CFDK-slab-replay-qualification-v1"
        and q["protocol_sha256"] == RULES == sha(root / "fixtures/slab-replay-protocol-v1.json")
        and q["plan_sha256"] == PLAN_PIN
        and sha(reports / "slab-replay-qualification.json")
        == sha(recovery / "qualification.json"),
        "different slab campaign or protocol",
    )
    need(
        q["historical_reports"] == REFERENCE
        and all(sha(reports / n) == v for n, v in REFERENCE.items())
        and set(q["source_sha256"]) == {"slab_replay.py", "qualify_slab_replay.py", "common.py"}
        and all(sha(root / "scripts" / n) == v for n, v in q["source_sha256"].items()),
        "source or comparison baseline differs",
    )
    native = json.loads((full / "build-repro.json").read_bytes())
    hd = json.loads((full / "walnut-full.json").read_bytes())
    dense = root / "data/full-corrected-512.f32"
    need(
        native["identical"] is True
        and native["rebuilds"] == 2
        and q["native_products"]
        == d["products"]
        == {"cone-fdk.exe": native["executable_sha256"], "kernels.ptx": native["ptx_sha256"]}
        and all(sha(root / "build" / n) == v for n, v in d["products"].items()),
        "wrong native products",
    )
    need(
        hd["views"] == 720
        and hd["offline_subset_bitwise_equal"] is True
        and hd["duplicate_360_excluded"] is True
        and hd["output_sha256"] == d["volume_sha256"] == q["original_volume_sha256"] == sha(dense)
        and sha(root / "data/full-corrected.geom") == d["geometry_sha256"]
        and sha(root / "data/full-corrected.f32") == d["projections_sha256"],
        "fresh measured scan differs",
    )
    cold, resumed = [recovery / n for n in ("cold", "resumed")]
    for dest in (cold, resumed):
        need(
            verify_volume(pp, PLAN_PIN, dest)["volume_sha256"] == d["volume_sha256"]
            and sha(dest / "manifest.json")
            == q["cold"]["manifest_sha256"]
            == q["resumed"]["manifest_sha256"],
            "delivery mismatch",
        )
    need(
        compare_outputs(dense, [cold, resumed], d) == q["slabs_compared"]
        and q["directly_compared_voxels"] == 512**3,
        "different full-grid slabs",
    )
    need(
        q["interrupted"]
        == dict(
            exit_code=73,
            complete_output_present=False,
            committed_z_and_bytes=[[s["z"], size(d, s)] for s in d["slabs"][:2]],
        ),
        "crash observation",
    )
    for tag, start, cache in [("cold", 0, "cold-cache"), ("resumed", 2, "resume-cache")]:
        v = q[tag]
        need(
            v["volume_sha256"] == d["volume_sha256"]
            and v["reused"] == start
            and v["computed"] == 5 - start
            and v["reused_bytes"] == sum(size(d, x) for x in d["slabs"][:start])
            and len(v["native"]) == 5 - start
            and q["cache_bytes"][tag] == (recovery / cache / "slabs.sqlite3").stat().st_size,
            "wrong job evidence",
        )
        for s, stats in zip(d["slabs"][start:], v["native"], strict=True):
            need(
                stats["gpu"] == prior["gpu"]
                and stats["z"] == s["z"]
                and stats["depth"] == s["depth"]
                and 0 < stats["peak_owned_bytes"] <= d["budget_mib"] << 20,
                "different GPU or checkpoint",
            )
    publish(
        out,
        dict(
            scope="Original CT and interface regression plus real-process recovery within the measured walnut. Both restored deliveries compared directly with a fresh 720-view dense volume. Not a new metrology, disk throughput or power-loss result.",
            gpu=prior["gpu"],
            pytest=prior["pytest"],
            slab_tests=len(tests),
            interface_campaign_sha256=prior["campaign_sha256"],
            qualification_sha256=sha(recovery / "qualification.json"),
            plan_sha256=PLAN_PIN,
            protocol_sha256=RULES,
            volume_sha256=d["volume_sha256"],
            measured_views=720,
            directly_compared_voxels=512**3,
            slabs=5,
            reused_after_hard_exit=2,
            recomputed_after_hard_exit=3,
        ),
    )
    print(
        "PASS: full measured reconstruction, 5 trusted slabs and both deliveries directly rechecked"
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
        "parcels",
        "roi",
        "build",
        "trial",
        "campaign",
        "recovery",
        "out",
    ):
        p.add_argument(n, type=Path)
    verify(**vars(p.parse_args()))
