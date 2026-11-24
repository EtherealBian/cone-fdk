"""Retrospective development comparison (A-H) and strict, source-backed replay of E."""

import argparse
import csv
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
from audit import replay  # noqa: E402
from freeze import DEV, identity, token  # noqa: E402
from validate import ORIGIN, SOURCES, final_gates  # noqa: E402

EVID = ROOT / "pores/evidence"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact_tail(rows):
    """Linear 95th percentile from native integer volumes, with no rounded inputs."""
    errors = []
    pid, rid = set(), set()
    for row in rows:
        p, r = (int(row[k]) for k in ("pred_voxels", "ref_voxels"))
        a, b = (int(row[k]) for k in ("pred_id", "ref_scipy_label"))
        require(
            min(a, b) > 0
            and a not in pid
            and b not in rid
            and min(p, r) >= 64
            and 0.5 <= float(row["iou"]) <= 1,
            "Invalid eligible one-to-one match",
        )
        pid.add(a)
        rid.add(b)
        errors.append(Fraction(100 * abs(p - r), r))
    require(errors, "An empty match set is not a score")
    errors.sort()
    n, remainder = divmod(95 * (len(errors) - 1), 100)
    return (errors[n] * (100 - remainder) + errors[min(n + 1, len(errors) - 1)] * remainder) / 100


def compare(evid=EVID):
    candidates = {}
    worst = {}
    for model in "ABCDEFGH":
        directory = evid / ("candidate-" + model.lower())
        specimens, tails = {}, []
        for i in DEV:
            path = directory / f"sample{i}.json"
            pairs = directory / f"sample{i}-matches.csv"
            doc = read(path)
            require(
                doc["role"] == "development"
                and doc["method"].endswith(" " + model)
                and doc["specimen"] == i
                and doc["origin_xyz"] == list(ORIGIN)
                and doc["calibration_z"] == [160, 184]
                and doc["evaluation_z"] == [400, 528]
                and doc["native"]["shape_zyx"] == [128, 256, 256]
                and all(
                    doc["source"][k] == SOURCES[f"{k}/set1sample{i}{k}.zip"]
                    for k in ("raw", "segmented")
                ),
                "Incompatible source, method or ROI",
            )
            with pairs.open(encoding="utf8", newline="") as f:
                rows = list(csv.DictReader(f))
            tail = exact_tail(rows)
            o = doc["objects"]
            require(
                len(rows) == o["matches"]
                and math.isclose(
                    float(tail), o["p95_abs_volume_error_pct"], rel_tol=0, abs_tol=1e-10
                ),
                "Archived volume tail disagrees with integer volumes",
            )
            require(
                o["criterion"]
                == "both interior and >=64 voxels; descending IoU >=0.5, unique matches"
                and all(
                    0 < len(rows) <= o[f"eligible_{side}"]
                    and math.isclose(
                        o[f"matched_fraction_{side}"],
                        len(rows) / o[f"eligible_{side}"],
                        rel_tol=0,
                        abs_tol=1e-12,
                    )
                    for side in ("prediction", "reference")
                ),
                "Inconsistent matching rule or coverage denominator",
            )
            gates = final_gates(doc["metrics"], o)
            require(
                gates == doc["final_gates"]
                and len(gates) == 11
                and all(gates.values()) == doc["final_passed"],
                "Gate mismatch",
            )
            specimens[str(i)] = dict(
                p95_exact_pct=str(tail),
                matches=len(rows),
                eligible_prediction=o["eligible_prediction"],
                eligible_reference=o["eligible_reference"],
                failed=[k for k, v in gates.items() if not v],
                summary_sha256=digest(path),
                matches_sha256=digest(pairs),
            )
            tails.append(tail)
        worst[model] = max(tails)
        candidates[model] = dict(worst_exact_pct=str(max(tails)), specimens=specimens)
    winners = [m for m, p in worst.items() if p == min(worst.values())]
    return dict(
        selection="post_hoc_minimize_worst_specimen_matched_volume_p95",
        minimizers=winners,
        candidates=candidates,
    )


def check_replay(run, cache, evid=EVID):
    published = evid / "candidate-e"
    require(run.resolve() != published.resolve(), "Provide separately regenerated outputs")
    expected = read(published / "pipeline-identity.json")
    require(
        identity("E") == expected, "Historical implementation or dependency identity mismatch"
    )
    results = {}
    for i in DEV:
        name = f"sample{i}"
        a = read(published / (name + ".json"))
        b = read(run / (name + ".json"))
        require(
            a["pipeline_id"] == b["pipeline_id"] == token(expected),
            "Unexpected E pipeline identity",
        )
        for j in (a, b):
            del j["native"]["elapsed_s"]
        require(a == b, "Report mismatch beyond elapsed time")
        hashes = {}
        for ext in ("-matches.csv", "-pores.csv"):
            d = digest(run / (name + ext))
            require(
                d == digest(published / (name + ext)), f"Archived table differs: {name + ext}"
            )
            hashes[ext] = d
        require(
            digest(run / (name + "-review.html")) == a["review_sha256"],
            "Archived full viewer differs",
        )
        method, gates = replay(run, i, cache)
        require(
            method == a["method"] and gates == a["final_gates"],
            "Independent score replay differs",
        )
        results[str(i)] = dict(
            table_hashes=hashes,
            review_sha256=a["review_sha256"],
            scipy_rows=a["same_mask_scipy_rows"],
            matches=a["objects"]["matches"],
            failed=[k for k, v in gates.items() if not v],
        )
    return dict(
        pipeline_id=token(expected), ignored_report_fields=["native.elapsed_s"], specimens=results
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regenerated", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    data = compare()
    data["source_replay_e"] = check_replay(args.regenerated, args.cache)
    with args.out.open("x", encoding="utf8", newline="\n") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
