"""Export exactly the externally pinned commits of a real Spark queue, never a loose attempt."""

import argparse
import json
import struct
from pathlib import Path

import physical_queue as queue
import review_physical_snapshot as offline
from spark_publication import fresh, publish, staging

rw = queue.pb.rw
need = rw.need


def export(state, target, entries, pine=None):
    state = state.absolute()
    target = fresh(target)
    need(
        not state.is_relative_to(target) and not target.is_relative_to(state),
        "destination intersects queue",
    )
    pins = {k: rw.pin(p) for k, p in entries}
    need(
        len(pins) == len(entries) and set(pins) <= set(queue.CASES), "unknown or duplicate commit"
    )
    # Reauthenticate through the unchanged driver, including its input/worker lineage and numerical gates.
    status = queue.execute(state, pine)
    by_case = {r["case"]: r for r in status["cases"] if r["execution"] == "reused"}
    need(set(by_case) == set(pins), "external pins must exactly cover the committed cases")
    pack = bytearray(offline.MAGIC)

    def append(path, pin, limit, exact=None):
        b = rw.bounded(path, limit, pin, exact)
        need(len(pack) + 4 + len(b) <= offline.LIMIT, "snapshot exceeds limit")
        pack.extend(struct.pack("<I", len(b)))
        pack.extend(b)
        return b

    with queue.locked(state):
        # A worker might have committed between these two locks. Never publish a mixed snapshot.
        need(
            queue.owned_json(state / "identity.json") == queue.ident()
            and {p.name for p in (state / "commits").iterdir()} == {k + ".json" for k in pins},
            "queue changed",
        )
        append(queue.pb.qr.ROOT / "fixtures/physical-residual-batch-v1.json", offline.PLAN, 16384)
        if "walnut720" in pins:
            for name, key in [
                ("walnut-gb10-001", offline.NATIVE),
                ("walnut-astra-gb10-001", offline.ASTRA),
            ]:
                append(queue.M / name / "result.json", key, 2 << 20)
        has_pine = bool(set(pins) & {"pine90", "pine720"})
        need(has_pine == (pine is not None), "missing or unused pine parent")
        if has_pine:
            append(queue.M / "pine-gb10-001/qualification.json", pine, 2 << 20)
        for k in queue.CASES:
            if k not in pins:
                continue
            cfile = state / "commits" / (k + ".json")
            rec = queue.owned_json(cfile)
            r = by_case[k]
            need(
                rec["attempt"] == r["attempt"] and rec["report_sha256"] == r["report_sha256"],
                "different result since authentication",
            )
            append(cfile, pins[k], 16384)
            src = state / "attempts" / r["attempt"] / "result"
            q = json.loads(append(src / "qualification.json", r["report_sha256"], 2 << 20))
            nx, ny, _ = q["shape_xyz"]
            size = nx * ny * 4
            plane = q["planes"][r["evidence"]["aggregate"]["first_maximum_xyz"][2]]
            for role in ("candidate", "reference"):
                append(src / (role + "-slice.f32"), plane[role + "_sha256"], size, size)
    catalog = dict(
        format="CFDK-physical-snapshot-v1",
        plan_sha256=offline.PLAN,
        pine_parent_sha256=pine,
        case_commits=pins,
        pack_sha256=rw.hashed(pack),
    )
    raw = queue.encoded(catalog)
    catalog_pin = rw.hashed(raw)
    with staging(target, ".physical-snapshot-") as t:
        dest = Path(t) / "ready"
        dest.mkdir()
        (dest / "catalog.json").write_bytes(raw)
        (dest / "planes.cfdk").write_bytes(pack)
        checked = offline.review(dest, catalog_pin)
        for row in checked["cases"]:
            if row["case"] in pins:
                need(
                    queue.pb.same(
                        row["evidence"]["volume_aggregate"],
                        by_case[row["case"]]["evidence"]["aggregate"],
                    ),
                    "detached aggregate differs from qualified queue",
                )
        publish(dest, target)
    return dict(
        catalog_sha256=catalog_pin,
        pack_sha256=catalog["pack_sha256"],
        pack_bytes=len(pack),
        committed_cases=len(pins),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("state", type=Path)
    p.add_argument("target", type=Path)
    p.add_argument(
        "--commit", nargs=2, action="append", default=[], metavar=("CASE", "COMMIT_SHA256")
    )
    p.add_argument("--pine-parent")
    a = p.parse_args()
    try:
        print(json.dumps(export(a.state, a.target, a.commit, a.pine_parent)))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RuntimeError,
        OverflowError,
    ) as e:
        p.exit(1, f"snapshot export refused: {e}\n")


if __name__ == "__main__":
    main()
