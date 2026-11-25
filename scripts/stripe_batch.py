"""Resume long-grid, two-anchor reviews by specimen; never serve an unvalidated previous result."""

import argparse
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from anchor_audit import pin
from edge_review import MAPS, ROLES, check, scenarios
from region_batch import STATES, atomic, encode, exclusive, fingerprint, refused
from region_review import grid_header, read_bound
from stripe_review import review

MAX_REPORT = 2 << 20
CHUNK = 1 << 20
RESULT = "CFDK-striped-region-summary-v1"
PLAN = "CFDK-striped-batch-plan-v1"
INDEX = "CFDK-striped-batch-v1"


def engine():
    info = fingerprint()
    for n in ("stripe_batch.py", "stripe_review.py", "stripe_frontier.py", "anchor_audit.py"):
        info["sources_sha256"][n] = hashlib.sha256(
            Path(__file__).with_name(n).read_bytes()
        ).hexdigest()
    return info


def parse(recipe, out):
    raw, digest = read_bound(recipe, 1 << 20)
    d = json.loads(raw)
    check(
        isinstance(d, dict)
        and set(d) == {"format", "entries"}
        and d["format"] == PLAN
        and isinstance(d["entries"], list)
        and 1 <= len(d["entries"]) <= 1024,
        "invalid striped batch plan",
    )
    check(not recipe.is_relative_to(out), "recipe inside output")
    rows, ids = [], set()
    for e in d["entries"]:
        check(
            isinstance(e, dict) and e.get("kind") in ("edge", "witness-refusal"),
            "invalid entry kind",
        )
        edge = e["kind"] == "edge"
        check(
            set(e)
            == (
                {"id", "job", "kind", "manifest_sha256"}
                | ({"anchors_row_col", "band_rows"} if edge else set())
            ),
            "invalid entry fields",
        )
        ident = e["id"]
        check(
            isinstance(ident, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", ident)
            and ident.casefold() not in ids,
            "invalid or duplicate entry id",
        )
        ids.add(ident.casefold())
        pin(e["manifest_sha256"])
        check(isinstance(e["job"], str) and e["job"].strip(), "empty source path")
        if edge:
            a = e["anchors_row_col"]
            check(
                isinstance(a, list)
                and len(a) == 2
                and a[0] != a[1]
                and all(
                    isinstance(p, list)
                    and len(p) == 2
                    and all(type(x) is int and 0 <= x < 4096 for x in p)
                    for p in a
                ),
                "invalid material anchors",
            )
            check(
                type(e["band_rows"]) is int and 1 <= e["band_rows"] <= 4096, "invalid band rows"
            )
        path = (recipe.parent / e["job"]).resolve()
        check(
            not path.is_relative_to(out) and not out.is_relative_to(path),
            "source overlaps output",
        )
        rows.append((e, path))
    return digest, rows


def source_bytes(job, expected):
    """Identical bytes suffice to reuse a *committed* topology calculation. Never allocate the full grid."""
    raw, _ = read_bound(job / "manifest.json", 1 << 20, pin(expected))
    d = json.loads(raw)
    check(d["format"] == "CFDK-edge-review-v1", "not an edge review")
    hashes = d["outputs_sha256"]
    header, _ = read_bound(job / "material.grid", 4096, pin(hashes["material.grid"]))
    (h, w), *_ = grid_header(header)
    k = len(scenarios(d["levels_per_mm"], d["nominal_level_per_mm"]))
    sizes = {f"{role}-{i:02}.wall": 32 for role in ROLES for i in range(k)}
    sizes.update(dict(zip(MAPS, (1, 8, 8), strict=True)))
    total = 0
    for name, b in sizes.items():
        expected_hash = pin(hashes[name])
        size = h * w * b
        with (job / name).open("rb") as f:
            check(os.fstat(f.fileno()).st_size == size, "source size changed: " + name)
            dig = hashlib.sha256()
            left = size
            while left:
                buf = f.read(min(left, CHUNK))
                check(bool(buf), "short source: " + name)
                dig.update(buf)
                left -= len(buf)
            check(
                not f.read(1) and dig.hexdigest() == expected_hash,
                "source SHA-256 mismatch: " + name,
            )
        total += size
    return total


def describe(e, doc, address):
    row = dict(
        id=e["id"],
        kind=e["kind"],
        source_manifest_sha256=e["manifest_sha256"],
        result_sha256=address,
        result_file="results/" + address + ".json",
    )
    if e["kind"] == "witness-refusal":
        check(
            doc["format"] == "CFDK-upstream-refusal-v1"
            and doc["witness_manifest_sha256"] == e["manifest_sha256"],
            "wrong refusal checkpoint",
        )
        return dict(
            **row,
            status="upstream_refused",
            reason=doc["reason"],
            rejected_scan=doc["rejected_scan"],
        )
    r, work = doc["result"], doc["work"]
    check(
        doc["format"] == RESULT
        and r["edge_manifest_sha256"] == e["manifest_sha256"]
        and r["anchors_row_col"] == e["anchors_row_col"]
        and work["band_rows"] == e["band_rows"],
        "wrong striped checkpoint",
    )
    rows = r["scenarios"]
    incomplete = sum(x["unknown_cells"] > 0 for x in rows)
    return dict(
        **row,
        status="incomplete_coverage" if incomplete else "complete_coverage",
        anchor_states=dict(sorted(Counter(s["anchor_state"] for s in rows).items())),
        incomplete_pairs=incomplete,
        pairs=len(rows),
    )


def run(recipe, output, *, resume=False):
    recipe, out = Path(recipe).resolve(), Path(output).resolve()
    digest, rows = parse(recipe, out)
    identity = engine()
    ed = hashlib.sha256(encode(identity)).hexdigest()
    header = encode(
        dict(format="CFDK-striped-checkpoint-v1", recipe_sha256=digest, engine=identity)
    )
    journal = out / "striped.sqlite3"
    if resume:
        check(
            journal.is_file() and (out / "results").is_dir(), "not an initialized striped batch"
        )
    else:
        out.mkdir(parents=True, exist_ok=False)
    with exclusive(out), closing(sqlite3.connect(journal, timeout=0)) as db:
        if not resume:
            (out / "results").mkdir()
            with db:
                db.execute(
                    "CREATE TABLE identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), header BLOB NOT NULL)"
                )
                db.execute("CREATE TABLE results (id TEXT PRIMARY KEY, address TEXT NOT NULL)")
                db.execute("INSERT INTO identity VALUES(1,?)", (header,))
        else:
            check(
                db.execute("SELECT header FROM identity WHERE singleton=1").fetchone()
                == (header,),
                "recipe or engine changed; use a fresh batch",
            )
        (out / "manifest.json").unlink(missing_ok=True)
        entries = []
        computed = reused = verified = 0
        for e, folder in rows:
            key = e["manifest_sha256"]
            edge = e["kind"] == "edge"
            saved = db.execute("SELECT address FROM results WHERE id=?", (e["id"],)).fetchone()
            if saved:
                addr = pin(saved[0])
                raw, _ = read_bound(out / "results" / (addr + ".json"), MAX_REPORT, addr)
                cached = json.loads(raw)
                row = describe(e, cached, addr)
            try:
                if not edge:
                    current = refused(folder, key)
                elif saved:
                    count = source_bytes(folder, key)
                else:
                    current = review(folder, key, e["anchors_row_col"], e["band_rows"])
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as err:
                entries.append(
                    dict(
                        id=e["id"],
                        kind=e["kind"],
                        source_manifest_sha256=key,
                        status="input_error",
                        error_type=type(err).__name__,
                        reason=str(err),
                    )
                )
                continue
            if saved:
                if edge:
                    check(
                        count == cached["work"]["payload_bytes_read"],
                        "cached byte accounting differs",
                    )
                    verified += count
                else:
                    check(current == cached, "cached refusal differs")
                reused += 1
            else:
                data = encode(current)
                check(len(data) <= MAX_REPORT, "summary too large")
                addr = hashlib.sha256(data).hexdigest()
                target = out / "results" / (addr + ".json")
                if target.exists():
                    read_bound(target, MAX_REPORT, addr)
                else:
                    atomic(target, data)
                row = describe(e, current, addr)
                with db:
                    db.execute("INSERT INTO results VALUES(?,?)", (e["id"], addr))
                computed += 1
            entries.append(row)
        tally = Counter(e["status"] for e in entries)
        result = dict(
            format=INDEX,
            recipe_sha256=digest,
            engine_sha256=ed,
            counts={k: tally[k] for k in STATES},
            entries=entries,
        )
        atomic(out / "manifest.json", encode(result))
        return result, dict(
            computed=computed,
            reused=reused,
            revalidated_payload_bytes=verified,
            input_errors=tally["input_error"],
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--resume", action="store_true")
    try:
        result, execution = run(**vars(p.parse_args()))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        sqlite3.Error,
        OverflowError,
    ) as err:
        p.exit(1, f"striped batch failed: {err}\n")
    c = result["counts"]
    print(json.dumps(dict(counts=c, execution=execution)))
    raise SystemExit(
        1 if c["input_error"] else 2 if c["upstream_refused"] or c["incomplete_coverage"] else 0
    )
