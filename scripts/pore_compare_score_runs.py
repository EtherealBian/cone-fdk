"""Verify two complete measurement runs and publish their semantic and CSV digests."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from pore_axial_raw import DEV, sha, write


def compare(first, second, arms):
    if first.resolve() == second.resolve() or not arms or len(set(arms)) != len(arms):
        raise ValueError("Expected two separate runs and distinct arms")
    expected = {
        "summary.json",
        *(f"sample{i}.json" for i in DEV),
        *(f"sample{i}-{k}-{t}.csv" for i in DEV for k in arms for t in ("pores", "matches")),
    }
    if not all({p.name for p in folder.iterdir()} == expected for folder in (first, second)):
        raise ValueError("Incomplete or unexpected evidence population")
    tables = {}
    digests = {}
    pipeline = None
    for f in sorted(expected):
        p, q = first / f, second / f
        if f.endswith(".csv"):
            if p.read_bytes() != q.read_bytes():
                raise ValueError(f"Different table: {f}")
            tables[f] = sha(p)
            continue
        a, b = [deepcopy(json.loads(path.read_text(encoding="utf8"))) for path in (p, q)]
        if f != "summary.json":
            for d in (a, b):
                if set(d["arms"]) != set(arms):
                    raise ValueError("Changed arms")
                for v in d["arms"].values():
                    elapsed = v["native"].pop("elapsed_s")
                    if not isinstance(elapsed, (int, float)) or not elapsed >= 0:
                        raise ValueError("Invalid time")
        if a != b:
            raise ValueError(f"Non-timing differences: {f}")
        if pipeline is None:
            pipeline = a["pipeline_id"]
        if a["pipeline_id"] != pipeline:
            raise ValueError("Mixed pipelines")
        digests[f] = hashlib.sha256(
            json.dumps(a, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    return dict(
        pipeline_id=pipeline,
        arms=arms,
        byte_identical_tables=tables,
        semantic_json_sha256=digests,
        excluded_only="arms.*.native.elapsed_s",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("first", "second", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    a = p.parse_args()
    d = compare(a.first, a.second, a.arms)
    write(a.out, d)
    print(len(d["byte_identical_tables"]), "identical CSVs, 4 matching JSON documents")
