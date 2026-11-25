"""Reopen three frozen witnessed parents, compare every wall/map byte and every historical threshold pair.

Archive required. Parent measurements use simulated specimens; running the wall kernels
on a physical GPU does not make the specimens physical or validate dimensional accuracy.
"""

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import zlib
import numpy as np
from review_directory_masks import CASES, SHAPE, FORMAT, review as offline_review
from pathlib import Path
from build_edge_compat import ROOT, build
from spark_publication import fresh, publish, staging

OVERLAY = "8ce35c3262a7d95de292616a5daa58655c41e60cd72710501a8ead7a78adeeb7"
BASE = "7f4e16fd0e057af4d83dd551f4348317ef6f78977951c1d5344f105c3a214f23"
ORACLES = {
    "region_review.py": "535a624dd8bf287e31a65f2e3b275fb85b800135dd04b7661b421e77e695fe9d",
    "validate_edges.py": "d749599db9d48d7ac1029b1f3b8b1dda0c868d4449ddad0766d8e3e69df3c3c7",
}


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def same(a, b):
    with a.open("rb") as f, b.open("rb") as g:
        while chunk := f.read(1 << 20):
            if g.read(len(chunk)) != chunk:
                return False
        return not g.read(1)


def qualify(store, dest):
    from re import fullmatch

    store = Path(store).resolve(strict=True)
    dest = fresh(dest)
    need(not dest.is_relative_to(store), "output inside archive")
    overlay = ROOT / "build/spark/python-edge-directory-v1"
    build(overlay, OVERLAY)
    sys.path[:0] = [str(overlay), str(ROOT / "scripts")]
    import edge_review as single, edge_review_packed as packed, edge_job_inputs as adapter
    from validate_edges import raw_oracle
    from region_review import analyze, load

    for mod in (single, packed, adapter):
        need(Path(mod.__file__).parent == overlay, "overlay shadowed")
    for n, h in ORACLES.items():
        need(sha(ROOT / "scripts" / n) == h, "wrong oracle")

    def obj(h):
        need(isinstance(h, str) and fullmatch("[0-9a-f]{64}", h), "invalid object pin")
        p = store / h[:2] / h
        need(p == p.resolve(strict=True) and sha(p) == h, "wrong archived object")
        return p

    def dec(p):
        return json.loads(p.read_bytes())

    baseline = ROOT / "reports/packed-edge-qualification.json"
    need(sha(baseline) == BASE, "baseline changed")
    q = dec(baseline)
    need(set(q["cases"]) == set(CASES), "unexpected baseline cases")
    mask_bytes = bytearray()
    checks = {}
    refused = {}
    with (
        (ROOT.parent / "gpu.lock").open("rb") as lock,
        staging(dest, ".directory-edge-qual-") as temp,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX)
        outer = Path(temp)
        delivery = outer / "delivery"
        delivery.mkdir()
        for label, expected_levels in CASES.items():
            case = q["cases"][label]
            with tempfile.TemporaryDirectory(dir=outer) as t:
                t = Path(t).resolve()
                source = t / "parent"
                source.mkdir()
                products = t / "historical"
                products.mkdir()
                pp = case["parent_sha256"]
                manifest = obj(pp)
                parent = dec(manifest)
                (source / "manifest.json").symlink_to(manifest)
                for k in adapter.CAP:
                    p = source / k
                    p.parent.mkdir(exist_ok=True)
                    p.symlink_to(obj(parent["outputs_sha256"][k]))
                for k in ["cone-fdk.exe", "kernels.ptx"]:
                    (products / k).symlink_to(obj(parent["inputs_sha256"][k]))
                old = dec(obj(case["packed_edge_sha256"]))
                reg = dec(obj(case["packed_region_sha256"]))
                need(
                    reg["edge_manifest_sha256"] == case["packed_edge_sha256"],
                    "bad historical lineage",
                )
                levels = old["levels_per_mm"]
                pairs = [[i, j] for i in range(len(levels)) for j in range(len(levels))]
                need(
                    levels == expected_levels
                    and reg["levels_per_mm"] == levels
                    and reg["shape_ha"] == SHAPE
                    and [s["pair"] for s in reg["scenarios"]] == pairs,
                    "changed frozen threshold grid",
                )
                variants = {}
                for name, run, extra in [
                    ("single", single.run, {}),
                    ("packed", packed.run, {"companion_sha256": adapter.TOOLS["wall-pack.exe"]}),
                ]:
                    out = t / name
                    result = run(
                        source,
                        out,
                        levels,
                        budget=1,
                        parent_sha256=pp,
                        source_products=products,
                        **extra,
                    )
                    lease = result["input_lease"]
                    need(
                        lease["parent_anchor"] == "supplied"
                        and lease["separate_source_products"]
                        and lease["parent_products_sha256"]["cone-fdk.exe"]
                        != lease["executables_sha256"]["cone-fdk.exe"],
                        "false producer/runtime provenance",
                    )
                    need(
                        result["inputs_sha256"]["manifest.json"] == pp
                        and result["parent_projection_sha256"] == old["parent_projection_sha256"],
                        "wrong parent",
                    )
                    pins = result["outputs_sha256"]
                    need(pins == case["matched_outputs_sha256"], "different outputs")
                    bytes_compared = 0
                    for k, h in pins.items():
                        p = out / k
                        need(same(p, obj(h)), "different archived bytes: " + k)
                        bytes_compared += p.stat().st_size
                    actual_pin = sha(out / "manifest.json")
                    rr = analyze(out, 9.4, actual_pin)
                    need(
                        rr == dict(reg, edge_manifest_sha256=actual_pin)
                        and len(rr["scenarios"]) == len(pairs),
                        "different region report",
                    )
                    if name == "packed":
                        _, _, (ref, part), _, _ = load(out, actual_pin)
                        for i, j in pairs:
                            delta = ref[i] - part[j]
                            mask_bytes.extend(
                                np.where(np.isnan(delta), 2, delta >= rr["loss_gate_mm"])
                                .astype("u1")
                                .tobytes()
                            )
                    devices = (
                        {
                            s["gpu"]
                            for stages in result["stages"].values()
                            for s in stages
                            if s["source"] == "cuda_wall"
                        }
                        if name == "single"
                        else {v["native"]["gpu"] for v in result["packs"].values()}
                    )
                    need(devices == {"NVIDIA GB10"}, "wrong measured device")
                    oracle = raw_oracle(source, out, result)
                    variants[name] = {
                        "edge_manifest_sha256": actual_pin,
                        "maps_compared": len(pins),
                        "byte_compared": bytes_compared,
                        "native_calls": result["native_wall_calls"]
                        if name == "single"
                        else result["native_wall_pack_calls"],
                        "raw_cpu_oracle": oracle,
                    }
                checks[label] = {
                    "parent_sha256": pp,
                    "archived_packed_edge_sha256": case["packed_edge_sha256"],
                    "archived_region_sha256": case["packed_region_sha256"],
                    "levels_per_mm": levels,
                    "shape_ha": reg["shape_ha"],
                    "pairs": pairs,
                    "loss_gate_mm": reg["loss_gate_mm"],
                    "observed_flagged_cells": [
                        c["observed_flagged_cells"] for c in reg["scenarios"]
                    ],
                    "periodic_label_sha256": [c["labels_i32le_sha256"] for c in reg["scenarios"]],
                    "region_counts": [c["region_count"] for c in reg["scenarios"]],
                    "unknown_cells": [c["unknown_cells"] for c in reg["scenarios"]],
                    "results": variants,
                }
        for label, case in q["refused"].items():
            with tempfile.TemporaryDirectory(dir=outer) as t:
                t = Path(t).resolve()
                p = t / "parent"
                p.mkdir()
                (p / "manifest.json").symlink_to(obj(case["parent_sha256"]))
                out = t / "review"
                why = {}
                for name, run, kw in [
                    ("single", single.run, {}),
                    ("packed", packed.run, {"companion_sha256": adapter.TOOLS["wall-pack.exe"]}),
                ]:
                    try:
                        run(
                            p,
                            out,
                            [0.018, 0.02, 0.022],
                            parent_sha256=case["parent_sha256"],
                            source_products=p,
                            **kw,
                        )
                    except ValueError as err:
                        need(str(err) == "unsupported parent", "wrong refusal: " + str(err))
                        why[name] = str(err)
                    else:
                        raise AssertionError("unqualified parent accepted")
                    need(
                        not os.path.lexists(out) and not list(t.glob(".sealed-edge-*")),
                        "partial result after refusal",
                    )
                refused[label] = {"parent_sha256": case["parent_sha256"], "refusal": why}
        pack = delivery / "directory-edge-masks.zlib"
        pack.write_bytes(zlib.compress(mask_bytes, 9))
        report = dict(
            format=FORMAT,
            offline_masks=dict(
                format="CFDK-mask-u8-zlib-v1",
                case_order=list(CASES),
                sha256=sha(pack),
                uncompressed_bytes=len(mask_bytes),
                codes=["below", "observed", "unknown"],
            ),
            scope="Archived simulated witnessed jobs. Real GB10 CUDA execution and numerical replay, not physical dimensional accuracy.",
            overlay_sha256=OVERLAY,
            baseline_sha256=BASE,
            qualifier_sha256=sha(Path(__file__)),
            native_sha256=adapter.TOOLS,
            cases=checks,
            refused=refused,
        )
        (delivery / "qualification.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        receipt = delivery / "qualification.json"
        offline_review(receipt, sha(receipt), pack)
        publish(delivery, dest)
        return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("store", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    r = qualify(a.store, a.dest)
    print("accepted", list(r["cases"]), "refused", list(r["refused"]))
