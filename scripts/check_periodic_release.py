"""Join the saved exact-label trials, source-bound parcels and the actual non-CoreOnly CUDA run."""

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from anchor_audit import publish
from build_periodic_labels import sha
from check_bridge_regression import verify as bridge
from periodic_label_cpu import cpu
from qualify_periodic_labels import PROTOCOL, REFERENCES, checked, need, performance
from qualify_periodic_parcels import DEMOS, LEGACY, PINS, cases, crosswalk


def read(p):
    return json.loads(p.read_bytes())


def hardware(q, build, work):
    need(
        q["format"] == "CFDK-periodic-label-qualification-v1"
        and q["protocol_sha256"] == PROTOCOL
        and q["source_references"] == REFERENCES
        and q["build"] == read(build / "build.json")
        and q["build"]["identical_builds"] == 2,
        "protocol or build binding",
    )
    need(
        set(q["build"]["products"]) == {"periodic-labels.exe", "periodic-labels.ptx"}
        and all(sha(build / n) == v for n, v in q["build"]["products"].items()),
        "different native product",
    )
    timing = performance(q["pairs"])
    need(timing == q["timing"], "misreported speed decision")
    need(
        set(q["refusals"]) == {"existing", "budget", "truncated", "invalid_last_map"},
        "missing refusal control",
    )
    stats = q["workloads"]
    shapes = {
        "tiny": (729, 2, 3),
        "native": (40, 41, 360),
        "contention": (5, 257, 263),
        "single-row": (5, 1, 4096),
        "scale": (9, 1536, 960),
    }
    need(set(stats) == set(shapes), "study cases")
    for n, (k, h, w) in shapes.items():
        s = stats[n]
        need(
            s["shape_ha"] == [h, w]
            and s["maps"] == k
            and s["pair_cells"] == h * w * k
            and s["redzones_verified"] is True
            and s["peak_owned_bytes"] == (h * w + 3) // 4 * 4 + 8 * h * w + 1536 <= 32 << 20,
            "scope or owned bytes",
        )
        need(sha(work / (n + ".u8")) == s["input_sha256"], "different mask bytes")
        if n != "scale":
            need(
                sha(work / (n + ".i32"))
                == sha(work / (n + "-expected.i32"))
                == s["output_sha256"],
                "sample label mismatch",
            )
        else:
            need(
                s["repeated_gpu_output_sha256"] == [s["output_sha256"]] * 3,
                "different large trials",
            )
    with TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "independent-large.i32"
        cpu(work / "scale.u8", fresh, 1536, 960, 9)
        need(sha(fresh) == stats["scale"]["output_sha256"], "fresh full-scale CPU labels differ")
    return timing


def parcels(p, refs, qual, work):
    mixed = refs["stripe-batch-qualification.json"]
    need(
        p["qualification_sha256"] == sha(qual)
        and p["references"] == PINS
        and p["packed_crosswalk"] == {LEGACY: crosswalk(refs)}
        and p["recipe_sha256"] == mixed["recipe_sha256"],
        "parcel survey lineage",
    )
    need(set(p["cases"]) == set(mixed["entries"]), "missing or extra parcel source")
    total = maps = 0
    for n, e in mixed["entries"].items():
        r = p["cases"][n]
        need(r["status"] == e["status"], "changed source disposition")
        if e["kind"] == "witness-refusal":
            need(
                r == dict(status="upstream_refused", reason=e["reason"])
                and not (work / n).exists(),
                "parcel attached to a withheld input",
            )
            continue
        parent, sigs = cases(refs, n)
        rows = r["rows"]
        need(
            len(rows) == len(sigs) and {tuple(x["pair"]) for x in rows} == set(sigs),
            "duplicate or absent threshold pair",
        )
        states = Counter()
        for x in rows:
            i, j = x["pair"]
            path = work / n / f"{i}-{j}"
            d = checked(path / "manifest.json", x["manifest_sha256"])
            h, w = d["shape_ha"]
            need(
                d["edge_manifest_sha256"] == parent == e["source_manifest_sha256"]
                and d["qualification_sha256"] == p["qualification_sha256"]
                and d["pair"] == [i, j]
                and d["source_scenario"]["anchor_state"] == x["anchor_state"]
                and sha(path / "labels.i32")
                == d["certificate"]["labels_i32le_sha256"]
                == x["labels_sha256"]
                == sigs[i, j]
                and sha(path / "mask.u8")
                == d["certificate"]["mask_u8_sha256"]
                == x["mask_sha256"],
                "parcel bytes or native lineage mismatch",
            )
            states[x["anchor_state"]] += 1
            total += h * w
            maps += 1
        need(dict(states) == e["anchor_states"], "anchor distribution")
    need(total == p["pair_cells"] == 988920 and maps == p["maps"] == 67, "total observed scope")
    need(set(p["examples"]) == set(DEMOS.values()), "missing examples")
    for (n, i, j), name in DEMOS.items():
        d = p["examples"][name]
        z = work / f"periodic-{name}-demo.zip"
        need(
            d["case"] == n
            and d["pair"] == [i, j]
            and sha(work / n / f"{i}-{j}/manifest.json") == d["manifest_sha256"]
            and sha(z) == d["zip_sha256"]
            and z.stat().st_size == d["bytes"],
            "example not bound",
        )


def source_lists(build, qualification, survey):
    required = [
        (
            build,
            {
                "periodic_labels.cpp",
                "periodic_labels.cu",
                "files.hpp",
                "gpu.hpp",
                "build_periodic_labels.py",
            },
        ),
        (
            qualification,
            {
                "periodic_label_cpu.py",
                "qualify_periodic_labels.py",
                "qualify_stripe_scale.py",
                "qualify_anchors.py",
                "region_review.py",
            },
        ),
        (
            survey,
            {"periodic_certificate.py", "periodic_parcel.py", "qualify_periodic_parcels.py"},
        ),
    ]
    for group, names in required:
        need(set(group) == names, "missing or unexpected checkout source")


def verify(full, edge, regions, reports, build, study, survey, out):
    old = bridge(full, edge, regions, reports)
    need(
        old["pytest"]["tests"] >= 601
        and sum(old["pytest"][k] for k in ("skipped", "errors", "failures")) == 0,
        "missing full tests",
    )
    qp = study / "qualification.json"
    q = read(qp)
    p = read(survey / "result.json")
    need(q["gpu"] == old["gpu"], "GPU mismatch")
    root = full.parent
    source_lists(q["build"]["sources"], q["verification_sources"], p["sources"])
    for n, v in q["build"]["sources"].items():
        need(
            sha(root / ("scripts" if n.endswith(".py") else "src") / n) == v,
            "checkout build source mismatch",
        )
    for n, v in {**q["verification_sources"], **p["sources"]}.items():
        need(sha(root / "scripts" / n) == v, "checkout verifier source mismatch")
    t = hardware(q, build, study)
    refs = {n: checked(reports / n, v) for n, v in PINS.items()}
    parcels(p, refs, qp, survey)
    publish(
        out,
        dict(
            scope="Clean Windows repeat build and non-CoreOnly regression, including the original CUDA volumes and the packed/witness, bridge and specimen campaigns. Exact preclassified label backend and source-bound parcel verification are separate gates.",
            gpu=q["gpu"],
            pytest=old["pytest"],
            bridge_replay_sha256=old["bridge_replay_sha256"],
            prior_region_regression_sha256=old["prior_region_regression_sha256"],
            qualified_products=q["build"]["products"],
            protocol_sha256=PROTOCOL,
            qualification_sha256=sha(qp),
            parcel_campaign_sha256=sha(survey / "result.json"),
            verified_pair_cells=p["pair_cells"],
            timing=t,
        ),
    )


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    for n in ("full", "edge", "regions", "reports", "build", "study", "survey", "out"):
        arg.add_argument(n, type=Path)
    verify(**vars(arg.parse_args()))
