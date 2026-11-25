"""Serial, resumable region inspection of pinned offline reviews and upstream refusals."""

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing, contextmanager
from importlib.metadata import version
from pathlib import Path

from edge_review import ROLES, check
from region_review import analyze, load, read_bound

HEX = re.compile("[0-9a-f]{64}")
MAX_RESULT = 64 << 20
STATES = ("complete_coverage", "incomplete_coverage", "upstream_refused", "input_error")


def encode(obj):
    return (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode("utf8")


def fingerprint():
    here = Path(__file__).resolve().parent
    files = {
        n: hashlib.sha256((here / n).read_bytes()).hexdigest()
        for n in (
            "region_batch.py",
            "region_review.py",
            "edge_review.py",
            "wall.py",
            "remount_loss.py",
        )
    }
    return dict(
        python=sys.version, numpy=version("numpy"), scipy=version("scipy"), sources_sha256=files
    )


def atomic(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def plan(recipe, output):
    raw, digest = read_bound(recipe, 1 << 20)
    doc = json.loads(raw)
    check(
        isinstance(doc, dict)
        and doc.get("format") == "CFDK-region-batch-plan-v1"
        and set(doc) == {"format", "entries"},
        "invalid batch recipe",
    )
    entries = doc["entries"]
    check(isinstance(entries, list) and 1 <= len(entries) <= 1024, "supply 1..1024 entries")
    check(not recipe.is_relative_to(output), "recipe must be outside the batch output")
    used = set()
    rows = []
    for e in entries:
        check(isinstance(e, dict), "invalid entry")
        kind = e.get("kind")
        ident = e.get("id")
        key = e.get("manifest_sha256")
        loc = e.get("job")
        check(
            kind in ("edge", "witness-refusal")
            and set(e)
            == (
                {"id", "kind", "job", "manifest_sha256"}
                | ({"area_radius_mm"} if kind == "edge" else set())
            ),
            "invalid entry kind or fields",
        )
        check(
            isinstance(ident, str)
            and re.fullmatch("[A-Za-z0-9][A-Za-z0-9._-]{0,63}", ident)
            and ident.casefold() not in used,
            "invalid or duplicate entry id",
        )
        used.add(ident.casefold())
        check(
            isinstance(key, str) and HEX.fullmatch(key) and isinstance(loc, str) and loc.strip(),
            "each source requires a manifest SHA-256 and job directory",
        )
        if kind == "edge":
            r = e["area_radius_mm"]
            check(type(r) in (float, int) and math.isfinite(r) and r > 0, "invalid area radius")
        folder = (recipe.parent / loc).resolve()
        check(
            not output.is_relative_to(folder) and not folder.is_relative_to(output),
            "batch and sources must be disjoint",
        )
        rows.append((e, folder))
    return digest, rows


@contextmanager
def exclusive(out):
    # A separate SQLite writer lease survives between case commits; process exit releases it.
    with closing(sqlite3.connect(out / "lease.sqlite3", timeout=0)) as db:
        try:
            db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as err:
            raise RuntimeError("cannot acquire batch writer lease") from err
        yield


def refused(folder, pin):
    data, _ = read_bound(folder / "manifest.json", 1 << 20, pin)
    m = json.loads(data)
    check(
        m["format"] == "CFDK-remount-witness-job-v1"
        and m["status"] == "datum_rejected"
        and m["loss"] is None
        and m["rejected_scan"] in ROLES
        and isinstance(m["reason"], str)
        and m["reason"]
        and "material-loss.json" not in m["outputs_sha256"],
        "not a pinned upstream witness refusal",
    )
    return dict(
        format="CFDK-upstream-refusal-v1",
        witness_manifest_sha256=pin,
        rejected_scan=m["rejected_scan"],
        reason=m["reason"],
    )


def cached_result(out, key):
    check(isinstance(key, str) and HEX.fullmatch(key), "invalid checkpoint address")
    raw, _ = read_bound(out / "results" / (key + ".json"), MAX_RESULT, key)
    return json.loads(raw)


def describe(e, obj, address):
    base = dict(
        id=e["id"],
        kind=e["kind"],
        source_manifest_sha256=e["manifest_sha256"],
        result_file=f"results/{address}.json",
        result_sha256=address,
    )
    if e["kind"] == "witness-refusal":
        check(
            obj["format"] == "CFDK-upstream-refusal-v1"
            and obj["witness_manifest_sha256"] == e["manifest_sha256"],
            "wrong cached witness refusal",
        )
        return dict(
            **base,
            status="upstream_refused",
            reason=obj["reason"],
            rejected_scan=obj["rejected_scan"],
        )
    check(
        obj["format"] == "CFDK-region-review-v1"
        and obj["edge_manifest_sha256"] == e["manifest_sha256"]
        and obj["area_radius_mm"] == e["area_radius_mm"],
        "wrong cached edge report",
    )
    return dict(
        **base,
        status="incomplete_coverage" if obj["incomplete_pair_count"] else "complete_coverage",
        scenario_count=len(obj["scenarios"]),
        incomplete_pair_count=obj["incomplete_pair_count"],
        observed_flagged_extrema=obj["observed_extrema"]["observed_flagged_cells"],
    )


def process(rows, out, db, recipe_sha, engine_sha):
    entries = []
    reused = computed = 0
    # No stale completed index is presented while sources are being rechecked.
    (out / "manifest.json").unlink(missing_ok=True)
    for e, folder in rows:
        ident, pin = e["id"], e["manifest_sha256"]
        saved = db.execute("SELECT address FROM results WHERE id=?", (ident,)).fetchone()
        if saved:
            obj = cached_result(out, saved[0])  # damaged checkpoint is not an input refusal
            entry = describe(e, obj, saved[0])
        try:
            if e["kind"] == "witness-refusal":
                current = refused(folder, pin)
            elif saved:
                load(folder, pin)  # re-hash and validate current source bytes, skip topology
            else:
                current = analyze(folder, e["area_radius_mm"], pin)
        except (ValueError, OSError, KeyError, TypeError, OverflowError) as err:
            entries.append(
                dict(
                    id=ident,
                    kind=e["kind"],
                    source_manifest_sha256=pin,
                    status="input_error",
                    error_type=type(err).__name__,
                    reason=str(err),
                )
            )
            continue
        if saved:
            if e["kind"] == "witness-refusal":
                check(current == obj, "cached upstream refusal differs")
            reused += 1
        else:
            raw = encode(current)
            check(len(raw) <= MAX_RESULT, "result exceeds checkpoint size limit")
            address = hashlib.sha256(raw).hexdigest()
            target = out / "results" / (address + ".json")
            if target.exists():
                read_bound(target, MAX_RESULT, address)
            else:
                atomic(target, raw)
            entry = describe(e, current, address)
            with db:
                db.execute("INSERT INTO results(id,address) VALUES(?,?)", (ident, address))
            computed += 1
        entries.append(entry)
    tally = Counter(r["status"] for r in entries)
    summary = dict(
        format="CFDK-region-batch-v1",
        recipe_sha256=recipe_sha,
        engine_sha256=engine_sha,
        counts={s: tally[s] for s in STATES},
        entries=entries,
    )
    atomic(out / "manifest.json", encode(summary))
    return summary, dict(computed=computed, reused=reused, input_errors=tally["input_error"])


def run(recipe, output, *, resume=False):
    recipe, out = Path(recipe).resolve(), Path(output).resolve()
    digest, rows = plan(recipe, out)
    engine = fingerprint()
    ed = hashlib.sha256(encode(engine)).hexdigest()
    header = encode(dict(format="CFDK-region-checkpoint-v1", recipe_sha256=digest, engine=engine))
    if resume:
        check(
            out.is_dir() and (out / "journal.sqlite3").is_file() and (out / "results").is_dir(),
            "not an initialized batch",
        )
    else:
        out.mkdir(parents=True, exist_ok=False)
    with exclusive(out), closing(sqlite3.connect(out / "journal.sqlite3", timeout=0)) as db:
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
                "recipe or engine identity changed; use a new batch",
            )
        return process(rows, out, db, digest, ed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("recipe", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--resume", action="store_true", help="revalidate sources and reuse committed cases"
    )
    try:
        log, stats = run(**vars(p.parse_args()))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        sqlite3.Error,
        OverflowError,
    ) as err:
        p.exit(1, f"batch failed: {err}\n")
    print(json.dumps(dict(counts=log["counts"], execution=stats)))
    raise SystemExit(
        1
        if log["counts"]["input_error"]
        else 2
        if (log["counts"]["upstream_refused"] or log["counts"]["incomplete_coverage"])
        else 0
    )
