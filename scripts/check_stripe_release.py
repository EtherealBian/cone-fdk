"""Bind a fresh striped replay to its dense oracle, pinned anchor evidence, and full CUDA regression."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from copy import deepcopy
from itertools import product
from pathlib import Path

from anchor_audit import publish
from check_bridge_regression import verify as bridge
from compare_native import sha
from validate_roi import require


def get(p):
    return json.loads(p.read_bytes())


def scale(record):
    """Compare deterministic values separately from OS working-set observations."""
    d = deepcopy(record)
    require(d["width"] == 960 and d["full_detail_dense_limit_cells"] == 1 << 20, "scale design")
    require(
        d["process_memory_gates"]
        == dict(max_peak_mib=256, max_growth_same_band_4x_height_mib=48),
        "memory policy",
    )
    require(set(d["cases"]) == {"384", "1536"}, "missing size")
    rss = {}
    for h in (384, 1536):
        group = d["cases"][str(h)]
        truth, trials = group["dense_oracle"], group["trials"]
        require(
            [tuple(x["pair"]) for x in truth] == list(product(range(3), repeat=2)),
            "missing oracle pair",
        )
        require([x["band_rows"] for x in trials] == [8, 32], "missing or reordered trial")
        for t in trials:
            rows = t["band_rows"]
            peak = t.pop("os_peak_working_set_bytes")
            require(type(peak) is int and 0 < peak < 256 << 20, "OS peak gate")
            require(
                t["bands"] * rows == h
                and t["pair_cells"] == 9 * h * 960
                and t["peak_band_cells"] == rows * 960
                and 0 < t["max_retained_components_per_frontier"] <= 960
                and t["payload_bytes_read"] == 209 * h * 960
                and t["hashed_payloads"] == 9,
                "work accounting",
            )
            rss[h, rows] = peak
    for rows in (8, 32):
        require(rss[1536, rows] - rss[384, rows] < 48 << 20, "OS growth gate")
    return d, {str(h): [rss[h, r] for r in (8, 32)] for h in (384, 1536)}


def compare_jobs(jobs, anchors, audits):
    require(
        set(jobs["cases"])
        == set(anchors["cases"])
        == {"open-control", "critical-a", "critical-b", "joined-control", "invalid-edge-control"},
        "job scope",
    )
    n = total = 0
    for name, case in jobs["cases"].items():
        p = audits / (name + ".json")
        require(sha(p) == anchors["cases"][name]["audit_sha256"], "anchor digest")
        a, r = get(p), case["result"]
        require(
            all(
                a[k] == r[k]
                for k in (
                    "anchors_row_col",
                    "levels_per_mm",
                    "shape_ha",
                    "nominal_pair",
                    "loss_gate_mm",
                )
            ),
            "material geometry or level mismatch",
        )
        outcome = [(s["pair"], s["anchor_state"]) for s in r["scenarios"]]
        require(
            outcome == [(s["pair"], s["state"]) for s in a["scenarios"]]
            and dict(Counter(v for _, v in outcome))
            == a["counts"]
            == anchors["cases"][name]["counts"],
            "stripe vs anchor outcome",
        )
        require([t["band_rows"] for t in case["trials"]] == [1, 7, 32], "production partitions")
        n += len(outcome)
        h, w = r["shape_ha"]
        total += h * w * len(outcome)
    require(
        n == jobs["maps"] == anchors["maps"] == 40
        and total == jobs["pair_cells"] == anchors["pair_cells"] == 590400,
        "pair scope",
    )
    return dict(maps=n, pair_cells=total)


def verify(full, edge, regions, reports, replay):
    parent = bridge(full, edge, regions, reports)
    require(parent["pytest"]["tests"] >= 492, "missing striped full run")
    post = ET.parse(full / "stripe-final-tests.xml").find("testsuite").attrib
    suite = {k: int(post[k]) for k in ("tests", "failures", "errors", "skipped")}
    require(
        suite["tests"] >= 503 and not any(suite[k] for k in ("failures", "errors", "skipped")),
        "final replay tests",
    )
    jp, ap, sp = (
        replay / "jobs.json",
        replay / "anchors/qualification.json",
        replay / "scale/qualification.json",
    )
    checked = {}
    for file, path in (("stripe-qualification.json", jp), ("anchor-qualification.json", ap)):
        require(
            path.read_bytes() == (reports / file).read_bytes(),
            "deterministic qualification changed: " + file,
        )
        checked[file] = sha(path)
    archived, rss = scale(get(reports / "stripe-scale.json"))
    fresh, replay_rss = scale(get(sp))
    require(archived == fresh, "changed source, fixture, work or oracle")
    for record in (get(jp), get(ap), fresh):
        for name, digest in record["source_sha256"].items():
            require(sha(Path(__file__).with_name(name)) == digest, "unbound source: " + name)
    counts = compare_jobs(get(jp), get(ap), replay / "anchors")
    return dict(
        protocol="Two native builds and full non-CoreOnly regression. Historical packed, bridge and independent specimen campaigns retained. Separate isolated CPU replay of pinned native jobs and synthetic OS-memory gate. Final pytest repeated after hardening and release-verifier tests.",
        gpu=parent["gpu"],
        full_pytest=parent["pytest"],
        final_pytest=suite,
        prior_bridge_replay_sha256=parent["bridge_replay_sha256"],
        native_volume_regression_sha256=parent["prior_region_regression_sha256"],
        qualifications_sha256=checked,
        archived_scale_sha256=sha(reports / "stripe-scale.json"),
        replay_scale_sha256=sha(sp),
        archived_peak_bytes_8_32=rss,
        replay_peak_bytes_8_32=replay_rss,
        **counts,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("full", "edge", "regions", "reports", "replay", "out"):
        p.add_argument(n, type=Path)
    args = vars(p.parse_args())
    out = args.pop("out")
    publish(out, verify(**args))
    print(
        "PASS: full regression, production maps, portable anchor certificates, and memory gates"
    )
