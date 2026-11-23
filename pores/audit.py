"""Replay a candidate's offline evidence rather than trusting its recorded pass flag.

Default: both registered confirmation specimens (4,6), one method. Development
reports can be audited separately but never produce confirmation. This checks
numerical consistency, not external traceability or independence of test data.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pore_check import reference
from validate import ORIGIN, SOURCES, SPEC, final_gates, objects, score, sha, stack, verified


def require(check, message):
    if not check:
        raise ValueError(message)


def decode(triples, size, binary=False):
    out = np.zeros(size, "i8")
    end = 0
    for start, count, value in triples:
        require(
            all(type(x) is int for x in (start, count, value))
            and start >= end
            and count > 0
            and start + count <= size
            and 0 < value <= (1 if binary else 2**31 - 1),
            "Malformed label run",
        )
        out[start : start + count] = value
        end = start + count
    return out


def replay(directory, i, cache=None):
    stem = f"sample{i}"

    def get(suffix):
        return directory / (stem + suffix)

    report = json.loads(get(".json").read_text(encoding="utf8"))
    native = report["native"]
    spacing = (SPEC[i][2],) * 3
    require(
        report["specimen"] == i
        and report["origin_xyz"] == list(ORIGIN)
        and report["calibration_z"] == [160, 184]
        and report["evaluation_z"] == [400, 528],
        "Unexpected specimen or split",
    )
    for kind in ("raw", "segmented"):
        require(
            report["source"][kind] == SOURCES[f"{kind}/set1sample{i}{kind}.zip"],
            "Unexpected archive identity",
        )
    require(
        report["role"] == ("confirmation" if i in (4, 6) else "development"),
        "Unexpected specimen role",
    )
    for suffix, key in [("-pores.csv", "table_sha256"), ("-review.html", "review_sha256")]:
        require(sha(get(suffix)) == report[key], "Evidence digest mismatch: " + suffix)
    html = get("-review.html").read_text(encoding="utf8")
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    require(m is not None, "Missing review payload")
    p = json.loads(m[1])
    del html, m
    shape = (128, 256, 256)
    require(
        p["specimen"] == i
        and p["origin"] == list(ORIGIN)
        and p["spacing"] == spacing[0]
        and p["shape"] == list(shape)
        and len(p["layers"]) == shape[0],
        "Wrong viewer geometry",
    )
    ids = np.empty(shape, "i8")
    ref = np.empty(shape, bool)
    for z, layer in enumerate(p["layers"]):
        ids[z] = decode(layer["objects"], 65536).reshape(256, 256)
        ref[z] = decode(layer["reference"], 65536, True).reshape(256, 256)
    if cache is not None:
        # Pin the replay to the authenticated archives and independently reload test labels.
        verified(cache, i, "raw", False)
        seg, _ = verified(cache, i, "segmented", False)
        np.testing.assert_array_equal(ref, stack(seg, i, "segmented", 400, 528))
    require(
        native["spacing_xyz_um"] == list(spacing)
        and native["shape_zyx"] == list(shape)
        and native["connectivity"] == 6
        and [native["slab_depth"], report["second_slab_depth"]] == [8, 37],
        "Unexpected native geometry/partition",
    )
    mask = ids != 0
    lab, rows = reference(mask, 6, spacing, get("-pores.csv"), native)
    lookup = np.zeros(int(lab.max()) + 1, "i8")
    require(
        len(rows) == report["same_mask_scipy_rows"] == len(p["objects"]), "Object count mismatch"
    )
    for row in rows:
        xyz = tuple(int(row["seed_" + axis]) for axis in "zyx")
        key = row["id"]
        lookup[lab[xyz]] = int(key)
        require(p["objects"][key]["row"] == row, "Viewer row mismatch")
    np.testing.assert_array_equal(ids, lookup[lab])
    metrics = score(mask, ref)
    with TemporaryDirectory() as tmp:
        matches = Path(tmp) / "matches.csv"
        stats = objects(lab, ref, rows, matches)
        with matches.open(newline="") as x, get("-matches.csv").open(newline="") as y:
            require(list(csv.reader(x)) == list(csv.reader(y)), "Match table disagreement")
    gates = final_gates(metrics, stats)
    require(
        metrics == report["metrics"]
        and stats == report["objects"]
        and gates == report["final_gates"]
        and all(gates.values()) == report["final_passed"],
        "Recomputed scores disagree",
    )
    return report["method"], gates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--cache", type=Path, help="Required for confirmation; local NIST archives")
    ap.add_argument("--development", type=int, nargs="+", choices=[2, 3, 5])
    a = ap.parse_args()
    if not __debug__:
        ap.error("Do not run the audit with Python -O")
    if not a.development and a.cache is None:
        ap.error("Confirmation requires the authenticated source archives (--cache)")
    specimens = a.development or [4, 6]
    methods = set()
    ok = True
    for i in specimens:
        method, gates = replay(a.directory, i, a.cache)
        methods.add(method)
        fail = [k for k, passed in gates.items() if not passed]
        ok &= not fail
        print(f"Sample {i}: " + (", ".join(fail) if fail else "all numerical gates met"))
    require(len(methods) == 1, "Mixed candidate methods")
    print(
        "DEVELOPMENT ONLY — no release decision"
        if a.development
        else ("Confirmation numeric checks passed" if ok else "Confirmation FAILED")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
