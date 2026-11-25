"""Validate saved CUDA face planes, all offline ledgers, and the original CT regression together."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np
from anchor_audit import publish
from build_periodic_labels import sha
from check_periodic_roi_release import verify as previous
from periodic_interfaces import verify as offline
from qualify_interface_parcels import oracle
from qualify_periodic_interfaces import BUILD_NAMES, PROTOCOL, SCALE, account, direct, vector
from qualify_periodic_labels import need
from qualify_periodic_roi import CAMPAIGN


def read(path):
    return json.loads(path.read_bytes())


def source_set(root, group, expected):
    need(set(group) == set(expected), "unexpected or missing qualification source")
    for n, digest in group.items():
        need(sha(root / ("scripts" if n.endswith(".py") else "src") / n) == digest, n)


def hardware(root, build, work, scale, q):
    need(
        q["format"] == "CFDK-periodic-interface-qualification-v1"
        and q["protocol_sha256"]
        == PROTOCOL
        == sha(root / "fixtures/periodic-interface-protocol-v1.json"),
        "protocol",
    )
    b = q["build"]
    need(
        b == read(build / "build.json")
        and b["identical_builds"] == 2
        and set(b["products"]) == BUILD_NAMES
        and all(sha(build / n) == v for n, v in b["products"].items()),
        "binary evidence",
    )
    source_set(
        root,
        b["sources"],
        [
            "periodic_interfaces.cpp",
            "periodic_interfaces.cu",
            "gpu.hpp",
            "files.hpp",
            "build_periodic_interfaces.py",
            "build_periodic_labels.py",
        ],
    )
    source_set(
        root,
        q["sources"],
        ["qualify_periodic_interfaces.py", "interface_ledger.py", "qualify_periodic_labels.py"],
    )
    shapes = {
        "tiny": (729, 2, 3),
        "one-row": (4, 1, 7),
        "analytic": (5, 8, 17),
        "height-47": (2, 47, 83),
        "height-211": (2, 211, 83),
        "scale": (9, 1536, 960),
    }
    runs = q["workloads"]
    need(
        set(runs) == set(shapes)
        and set(q["refusals"]) == {"last", "short", "budget", "existing", "same_path"}
        and sha(scale) == SCALE,
        "native case scope",
    )
    for n, (k, h, w) in shapes.items():
        r = runs[n]
        src = scale if n == "scale" else work / (n + ".u8")
        need(
            r["shape_ha"] == [h, w]
            and r["maps"] == k
            and r["cells"] == k * h * w
            and r["input_sha256"] == sha(src)
            and len(r["runs"]) == 3,
            "native case or input",
        )
        oracle_hash = hashlib.sha256()
        with src.open("rb") as f:
            for _ in range(k):
                a = np.frombuffer(f.read(h * w), "u1").reshape(h, w)
                truth = vector(a)
                if n != "scale":
                    need(np.array_equal(direct(a), truth), "fresh pointwise interface mismatch")
                oracle_hash.update(truth.tobytes())
            need(not f.read(1), "trailing source mask")
        for band, d in zip((1, 7, 32), r["runs"], strict=True):
            account(d, h, w, k, band, 1)
            need(
                sha(work / f"{n}-b{band}.u8") == r["face_sha256"] == oracle_hash.hexdigest(),
                "face plane differs",
            )

    def peak(n):
        return [r["peak_owned_bytes"] for r in runs[n]["runs"]]

    need(peak("height-47") == peak("height-211"), "height scaled device memory")
    return peak("scale")


def survey(root, parcels, work, reports, p, qp):
    need(
        p["format"] == "CFDK-periodic-interface-campaign-v1"
        and p["protocol_sha256"] == PROTOCOL
        and p["device_qualification_sha256"] == qp
        and p["parcel_campaign_sha256"] == CAMPAIGN == sha(parcels / "result.json"),
        "source binding",
    )
    source_set(
        root,
        p["sources"],
        ["periodic_interfaces.py", "interface_ledger.py", "qualify_interface_parcels.py"],
    )
    old = read(parcels / "result.json")["cases"]
    need(set(old) == set(p["cases"]), "case scope")
    totals = np.zeros((4, 3), np.int64)
    maps = cells = refused = 0
    for name, c in old.items():
        v = p["cases"][name]
        need(v["status"] == c["status"], "case outcome")
        if v["status"] == "upstream_refused":
            need(not v["rows"] and not (work / name).exists(), "withheld source has faces")
            refused += 1
            continue
        indexed = {tuple(x["pair"]): x for x in v["rows"]}
        need(len(indexed) == len(c["rows"]) == len(v["rows"]), "pair count")
        for item in c["rows"]:
            i, j = item["pair"]
            row = indexed[i, j]
            sub = Path(name) / f"{i}-{j}"
            d = work / sub
            pp = item["manifest_sha256"]
            rp = sha(d / "manifest.json")
            need(
                row["result_manifest_sha256"] == rp
                and row["parcel_manifest_sha256"] == pp
                and read(d / "manifest.json")["qualification_sha256"] == qp,
                "result provenance",
            )
            m = offline(parcels / sub, d, pp, rp, 5)
            h, w = m["shape_ha"]
            counts = oracle(
                np.fromfile(parcels / sub / "mask.u8", "u1").reshape(h, w),
                np.fromfile(parcels / sub / "labels.i32", "<i4").reshape(h, w),
            )
            s = m["summary"]
            need(
                m["pair"] == [i, j]
                and counts == s["components"]
                and len(counts) == row["components"]
                and s["faces_by_direction_and_kind"] == row["faces"]
                and s["face_u8_sha256"] == row["face_sha256"]
                and s["observed_cells"] == row["observed_cells"],
                "per-component ledger differs",
            )
            totals += np.array(row["faces"], np.int64)
            maps += 1
            cells += h * w
    need(
        maps == p["maps"] == 67
        and cells == p["paired_cells"] == 988920
        and refused == 2
        and totals.tolist() == p["faces_by_direction_and_kind"],
        "campaign totals",
    )
    ex = p["example"]
    sub = Path("seam-flat-change/1-1")
    z = "interface-seam-demo.zip"
    need(
        ex["case"] == "seam-flat-change"
        and ex["pair"] == [1, 1]
        and sha(work / z) == sha(reports / z) == ex["zip_sha256"]
        and ex["result_manifest_sha256"] == sha(work / sub / "manifest.json")
        and offline(
            parcels / sub,
            work / "reopened-seam",
            ex["parcel_manifest_sha256"],
            ex["result_manifest_sha256"],
            3,
        )["summary"]
        == ex["summary"],
        "portable example differs",
    )


def verify(
    full, edge, regions, reports, backend, study, parcels, roi, build, trial, campaign, out
):
    with TemporaryDirectory() as temp:
        earlier = Path(temp) / "prior.json"
        previous(full, edge, regions, reports, backend, study, parcels, roi, earlier)
        parent = read(earlier)
    tests = [
        t
        for t in ElementTree.parse(full / "native-tests.xml").iterfind(".//testcase")
        if t.get("classname") == "tests.test_periodic_interfaces"
    ]
    need(
        len(tests) >= 25 and all(not list(t) for t in tests) and parent["pytest"]["tests"] >= 663,
        "missing or skipped interface controls",
    )
    qp = sha(trial / "qualification.json")
    cp = sha(campaign / "qualification.json")
    need(
        sha(reports / "interface-qualification.json") == qp
        and sha(reports / "interface-parcels.json") == cp,
        "archive differs",
    )
    q = read(trial / "qualification.json")
    p = read(campaign / "qualification.json")
    need(q["gpu"] == parent["gpu"], "mixed GPU")
    peaks = hardware(full.parent, build, trial, study / "scale.u8", q)
    survey(full.parent, parcels, campaign, reports, p, qp)
    publish(
        out,
        dict(
            scope="Original CT regression and qualified parcel replay. Exact typed discrete sides on simulated projection-derived sources and frozen masks; not physical perimeter, accuracy or timing.",
            pytest=parent["pytest"],
            interface_tests=len(tests),
            gpu=q["gpu"],
            protocol_sha256=PROTOCOL,
            device_qualification_sha256=qp,
            campaign_sha256=cp,
            roi_qualification_sha256=parent["qualification_sha256"],
            pressure_band_rows=[1, 7, 32],
            pressure_peak_owned_bytes=peaks,
            verified_maps=p["maps"],
            verified_cells=p["paired_cells"],
            faces_by_direction_and_kind=p["faces_by_direction_and_kind"],
            example_zip_sha256=p["example"]["zip_sha256"],
        ),
    )
    print(
        "PASS: full CT regression, three band heights per mask, 67 independent CPU ledger replays"
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
        "out",
    ):
        p.add_argument(n, type=Path)
    verify(**vars(p.parse_args()))
