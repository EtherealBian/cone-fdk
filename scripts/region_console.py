"""Export a pinned region batch into one static HTML with digest-checked, material-space maps."""

import argparse
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from edge_review import check
from region_batch import HEX, STATES, atomic, cached_result, describe, encode, plan, refused
from region_review import analyze, labels, load, read_bound

MAX_VIEWS = 8192
MAX_PAIR_CELLS = 8 << 20
MAX_SERIALIZED = 48 << 20
TEMPLATE = Path(__file__).with_name("region_console.html")


def packed_map(a, expected):
    a = a.astype("<i4", copy=False).ravel()
    raw = a.tobytes()
    check(hashlib.sha256(raw).hexdigest() == expected, "display labels differ from measured map")
    starts = np.r_[0, np.flatnonzero(a[:-1] != a[1:]) + 1]
    if 2 * len(starts) < a.size:
        runs = np.column_stack((np.diff(np.r_[starts, a.size]), a[starts])).astype("<i4")
        return ["rle32le", base64.b64encode(runs.tobytes()).decode("ascii")]
    return ["i32le", base64.b64encode(raw).decode("ascii")]


def export(recipe, batch, output, batch_sha256):
    recipe, batch, out = (Path(p).resolve() for p in (recipe, batch, output))
    check(isinstance(batch_sha256, str) and HEX.fullmatch(batch_sha256), "supply batch SHA-256")
    check(
        not out.exists() and not out.is_relative_to(batch) and not batch.is_relative_to(out),
        "use a new export folder outside the batch",
    )
    recipe_sha, rows = plan(recipe, out)
    index = json.loads(read_bound(batch / "manifest.json", 8 << 20, batch_sha256)[0])
    entries = index["entries"]
    states = Counter(x["status"] for x in entries)
    check(
        index["format"] == "CFDK-region-batch-v1"
        and index["recipe_sha256"] == recipe_sha
        and len(entries) == len(rows)
        and not (states.keys() - set(STATES))
        and {s: states[s] for s in STATES} == index["counts"],
        "batch recipe or counts differ",
    )
    result = []
    cells = views = used = 0
    for (e, folder), rec in zip(rows, entries, strict=True):
        pin = e["manifest_sha256"]
        check(
            (rec["id"], rec["kind"], rec["source_manifest_sha256"]) == (e["id"], e["kind"], pin),
            "batch source identity differs",
        )
        item = dict(id=e["id"], state=rec["status"], source_sha256=pin)
        result.append(item)
        if rec["status"] == "input_error":
            check(
                "result_sha256" not in rec and "result_file" not in rec,
                "input error has a result",
            )
            item["error_type"] = rec[
                "error_type"
            ]  # no local file paths; do not retry/upgrade stale errors
        else:
            key = rec["result_sha256"]
            obj = cached_result(batch, key)
            check(describe(e, obj, key) == rec, "batch entry differs from result")
            item["result_sha256"] = key
            if e["kind"] == "witness-refusal":
                check(refused(folder, pin) == obj, "changed witness refusal")
                item.update(reason=obj["reason"], rejected_scan=obj["rejected_scan"])
            else:
                h, w = obj["shape_ha"]
                count = len(obj["scenarios"])
                cells += h * w * count
                views += count
                check(
                    0 < cells <= MAX_PAIR_CELLS and views <= MAX_VIEWS,
                    "offline console size limit",
                )
                current = analyze(folder, e["area_radius_mm"], pin)
                check(encode(current) == encode(obj), "batch result differs from source replay")
                doc, _, (refs, parts), _, (_, _, dz) = load(folder, pin)
                item.update(
                    shape=[h, w],
                    levels=doc["levels_per_mm"],
                    nominal=obj["nominal_pair"],
                    gate=obj["loss_gate_mm"],
                    radius=obj["area_radius_mm"],
                    dz=dz,
                    scenarios=[],
                )
                for s in obj["scenarios"]:
                    i, j = s["pair"]
                    delta = refs[i] - parts[j]
                    lab = labels(delta >= item["gate"])
                    lab[np.isnan(delta)] = -1
                    item["scenarios"].append(
                        dict(**s, pixels=packed_map(lab, s["labels_i32le_sha256"]))
                    )
        used += len(json.dumps(item, allow_nan=False, separators=(",", ":")))
        check(used <= MAX_SERIALIZED, "offline console metadata limit")
    payload = dict(
        format="CFDK-region-console-v1",
        batch_sha256=batch_sha256,
        recipe_sha256=recipe_sha,
        entries=result,
    )
    template = TEMPLATE.read_text(encoding="utf8")
    check(template.count("__REGION_PAYLOAD__") == 1, "invalid template")
    # Never embed an external string as HTML; JSON is an inert data block. Escape raw end tags.
    data = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    html = template.replace("__REGION_PAYLOAD__", data).encode("utf8")
    out.mkdir(parents=True, exist_ok=False)
    atomic(out / "index.html", html)
    return dict(
        batch_sha256=batch_sha256,
        html_sha256=hashlib.sha256(html).hexdigest(),
        entries=len(result),
        maps=views,
        pair_cells=cells,
        html_bytes=len(html),
        counts=index["counts"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("recipe", "batch", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--batch-sha256", required=True)
    try:
        result = export(**vars(parser.parse_args()))
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as err:
        parser.exit(1, f"console export failed: {err}\n")
    print(json.dumps(result))
