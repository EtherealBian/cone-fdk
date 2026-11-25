"""Offline reviewer for explicit, committed physical cases; reads no paths from the payload."""

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

# The helper is shipped with the reviewer, never read from a submitted snapshot.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_witness as rw  # noqa: E402

need = rw.need
MAGIC = b"CFDKPS01"
LIMIT = 12 << 20
PLAN = "e8e263d1c02ab7782bed459d1195c528b35ef3fa7f1cdb07d22fa8fc576b4ed5"
NATIVE = "c1320694248460d2de39f892f377b1b3a7473d186f32d9971d789d45d140f0ec"
ASTRA = "fa5180be58a97e86e4be69c7515a6e8937b98c0a57ba2a3e6b402030a8aa4df4"
PINE = "4fd5f98f528e467532cefa8c498786e6e28fd662b2e09b4a92a8db77b4d69218"
PINE_SOURCE = "0e236a75a8d4ebe9dc09b1f1a79f8ab832c42df8738340d66b40ba13d4c5deb4"
PINE_TOOL = "0a17eebbcab53e6677e6b7528802a075868fd2240b9de546d5c43c4a3706400f"
WITNESS = "b127c8be1a94e96a24e9a1404e5923a7bfdd8b2a7cf034db4e6ee76f4d6bafef"
HARNESS = "2f21f3c7a2d517b5580567cd2feae38d5090a7958f420d67a92a2f0c735f350c"
BUILD = "b83e045b2e66612a48bde9bd92a0864ad26fc0bde08e5417b029478f0da6cd3c"


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def parse(data):
    def unique(pairs):
        obj = dict(pairs)
        need(len(obj) == len(pairs), "duplicate JSON key")
        return obj

    def reject(token):
        raise ValueError("nonfinite JSON number: " + token)

    return json.loads(data, object_pairs_hook=unique, parse_constant=reject)


class Frames:
    def __init__(self, data):
        need(data.startswith(MAGIC) and len(data) <= LIMIT, "pack header or size")
        self.data = data
        self.pos = len(MAGIC)

    def take(self, pin, limit, exact=None):
        rw.pin(pin)
        need(self.pos + 4 <= len(self.data), "missing frame")
        n = struct.unpack_from("<I", self.data, self.pos)[0]
        start = self.pos + 4
        end = start + n
        need(
            0 < n <= limit and (exact is None or n == exact) and end <= len(self.data),
            "frame length",
        )
        b = self.data[start:end]
        self.pos = end
        need(rw.hashed(b) == pin, "member digest")
        return b

    def doc(self, pin, limit=2 << 20):
        return parse(self.take(pin, limit))


def aggregate(planes, shape):
    """Fold reported plane statistics; this operation never reads a full volume."""
    nx, ny, nz = shape
    pixels = nx * ny
    rows = []
    need(type(planes) is list and len(planes) == nz, "missing planes")
    for i, p in enumerate(planes):
        need(type(p["z"]) is int and p["z"] == i, "plane order")
        rw.pin(p["candidate_sha256"])
        rw.pin(p["reference_sha256"])
        r = p["gpu"]
        need(
            type(r) is list
            and len(r) == 12
            and all(type(r[j]) is int for j in (0, 1, 2, 3, 4, 5, 7)),
            "plane counts",
        )
        need(
            r[0] == pixels
            and 0 <= r[1] <= pixels
            and pixels - r[1] >= r[2] >= r[3] >= r[4] >= r[5] >= 0
            and 0 <= r[7] < pixels,
            "plane ranges",
        )
        need(
            all(type(r[j]) in (int, float) and math.isfinite(r[j]) for j in (6, 8, 9, 10, 11))
            and min(r[6], *r[9:]) >= 0,
            "plane moments",
        )
        need(
            (r[6] == 0) == (r[1] == pixels)
            and all((r[j + 2] > 0) == (r[6] > c) for j, c in enumerate(rw.CUTS)),
            "peak/count disagreement",
        )
        rows.append(r)
    z = max(range(nz), key=lambda i: rows[i][6])
    v = rows[z]
    s, a, e, ref = (math.fsum(r[j] for r in rows) for j in range(8, 12))
    count = pixels * nz
    return dict(
        voxels=count,
        numeric_equal=sum(r[1] for r in rows),
        strict_exceedance=[sum(r[j] for r in rows) for j in range(2, 6)],
        max_abs=v[6],
        first_maximum_xyz=[v[7] % nx, v[7] // nx, z],
        signed_mean=s / count,
        mean_abs=a / count,
        rms=math.sqrt(e / count),
        rel_l2=math.sqrt(e / ref) if ref else None,
    )


def expected_source(case, parents, pine_pin):
    if case == "walnut720":
        n, a = parents["native"], parents["astra"]
        left = a["input_sha256"]["full-corrected-512.f32"]
        right = a["sha256"]
        need(
            a["passed"] is True
            and n["output_sha256"] == [left, left]
            and n["input_sha256"] == a["input_sha256"]
            and a["astra"]["astra_version"] == "2.5.0",
            "walnut parent disagreement",
        )
        dim = 512
        pins = {"native": NATIVE, "independent_astra": ASTRA}
        old = a["native_vs_spark_astra"]
    else:
        p = parents["pine"]
        need(
            p["format"] == "CFDK-Spark-physical-pine-qualification-v1"
            and p["protocol_sha256"] == PINE
            and p["source_manifest_sha256"] == PINE_SOURCE
            and p["script_sha256"] == PINE_TOOL
            and set(p["scans"]) == {"pine90", "pine720"},
            "pine parent identity",
        )
        r = p["scans"][case]
        dim = 256
        left = r["dense_sha256"]
        right = r["astra_sha256"]
        old = r["native_vs_astra"]
        pins = {"physical_qualification": pine_pin, "protocol": PINE}
        need(
            r["views"] == int(case[4:])
            and r["repeat_sha256"] == left
            and r["voxels"] == dim**3
            and len(r["rois"]) == 2,
            "pine case binding",
        )
    return dict(
        shape=[dim] * 3, case=case, parents=pins, pins=[rw.pin(left), rw.pin(right)], old=old
    )


def case_review(fr, case, commit_pin, src, pine_pin, profiles=None):
    commit = fr.doc(commit_pin, 16384)
    report_pin = rw.pin(commit["report_sha256"])
    att = commit["attempt"]
    need(
        type(att) is str
        and re.fullmatch(case + "-[a-f0-9]{32}", att)
        and same(
            commit,
            dict(
                format="CFDK-verified-case-commit-v1",
                case=case,
                attempt=att,
                report_sha256=report_pin,
                pine_parent=pine_pin,
            ),
        ),
        "case commit",
    )
    q = fr.doc(report_pin)
    shape = src["shape"]
    nx, ny, nz = shape
    need(
        q["format"] == "CFDK-paired-volume-qualification-v1"
        and q["protocol_sha256"] == rw.RULES
        and q["row_fields"] == rw.FIELDS
        and q["build_sha256"] == BUILD
        and q["harness_sha256"] == HARNESS
        and q["witness_tool_sha256"] == WITNESS
        and same(q["source"], src)
        and same(q["shape_xyz"], shape)
        and same(q["slices_verified_by_numpy"], nz),
        "wrong report or source",
    )
    totals = aggregate(q["planes"], shape)
    need(
        same(totals, q["aggregate"])
        and totals["max_abs"] == src["old"]["max_abs"]
        and totals["rel_l2"] is not None
        and math.isclose(totals["rel_l2"], src["old"]["rel_l2"], rel_tol=2e-13, abs_tol=0),
        "aggregate or parent baseline differs",
    )
    x, y, z = totals["first_maximum_xyz"]
    plane = q["planes"][z]
    size = 4 * nx * ny
    a, b = (fr.take(plane[k + "_sha256"], size, size) for k in ("candidate", "reference"))
    oracle = rw.scalar(a, b)
    rw.agree(plane["gpu"], oracle)
    if profiles is not None:
        profiles[case] = tuple(p["gpu"][6] for p in q["planes"])
    return dict(
        commit_sha256=commit_pin,
        report_sha256=report_pin,
        volume_sha256=src["pins"],
        source_parents=src["parents"],
        plane_rows_reaggregated=nz,
        peak_plane_voxels_remeasured=nx * ny,
        first_maximum_xyz=[x, y, z],
        max_abs=oracle[6],
        peak_plane_equal=oracle[1],
        peak_plane_strict_exceedance=oracle[2:6],
        volume_aggregate=totals,
    )


def decode(cat, data, *, profiles=None):
    need(
        set(cat) == {"format", "plan_sha256", "pine_parent_sha256", "case_commits", "pack_sha256"}
        and cat["format"] == "CFDK-physical-snapshot-v1"
        and cat["plan_sha256"] == PLAN
        and rw.hashed(data) == rw.pin(cat["pack_sha256"]),
        "snapshot identity",
    )
    fr = Frames(data)
    plan = fr.doc(PLAN, 16384)
    cases = plan["cases"]
    supplied = cat["case_commits"]
    names = [r["case"] for r in cases]
    pine = cat["pine_parent_sha256"]
    need(
        type(supplied) is dict
        and set(supplied) <= set(names)
        and len({rw.pin(v) for v in supplied.values()}) == len(supplied),
        "case set",
    )
    parents = {}
    if "walnut720" in supplied:
        parents.update(native=fr.doc(NATIVE), astra=fr.doc(ASTRA))
    if set(supplied) & {"pine90", "pine720"}:
        parents["pine"] = fr.doc(rw.pin(pine))
    else:
        need(pine is None, "unused pine parent")
    out = []
    for row in cases:
        k = row["case"]
        item = dict(row, status="not_submitted")
        if k in supplied:
            src = expected_source(k, parents, pine)
            item.update(
                status="committed",
                evidence=case_review(
                    fr, k, supplied[k], src, None if k == "walnut720" else pine, profiles
                ),
            )
        out.append(item)
    need(fr.pos == len(data), "trailing frames or bytes")
    return dict(
        format="CFDK-offline-physical-review-v1",
        cases=out,
        committed_cases=len(supplied),
        planned_cases=len(cases),
        specimens_with_evidence=len({r["specimen"] for r in out if r["status"] == "committed"}),
        planned_specimens=len({r["specimen"] for r in out}),
        all_cases_present=set(supplied) == set(names),
        scope="Same-grid native versus ASTRA. Reported planes are reaggregated; only the peak plane of each submitted case is remeasured from raw float32 pairs. No dimensional accuracy claim.",
    )


def review(folder, pin, *, profiles=None):
    cat = parse(rw.bounded(folder / "catalog.json", 16384, pin))
    return decode(
        cat, rw.bounded(folder / "planes.cfdk", LIMIT, cat["pack_sha256"]), profiles=profiles
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("catalog_sha256")
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    try:
        out = review(a.folder, a.catalog_sha256)
        if a.require_complete and not out["all_cases_present"]:
            p.exit(3, "missing committed physical cases\n")
        print(json.dumps(out, indent=2, allow_nan=False))
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OSError,
        OverflowError,
        RecursionError,
    ) as e:
        p.exit(1, f"snapshot refused: {e}\n")


if __name__ == "__main__":
    main()
