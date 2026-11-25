"""Audit a differential GB10 witness-to-interface campaign using only Python's standard library.

Supply the external campaign SHA and the original archived object store. No source tree,
CUDA card, or live job paths are required. This checks bytes and links, not physical truth.
"""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

CAMPAIGN = "1d0e7ec3923b388e48232e8c460cf1ce7c9f24f6d637a0cffe8d87d757047980"
FACES = "227b105a5890f3964a7b72bc439940ca39aa9ee607cc9355122928807c9e3ac6"
LABEL_Q = "dc2ae60f9757e6bfa334455a5874e1d0b03724abda7328c56549cd18a528d182"
FACE_Q = "1c30412f6097587c713c5384107a2dda3a65043fd4c29950f1a1fe86f3dfccd6"
ADAPTER = "22d215d0cf481b31972393f9c633431673aa04226385dcb21622e4516c94e28d"
PRODUCTS = {
    "wall-pack": "6351b4db3aa52e1df6a9651ccc329dcb238c45c6cdbc498dbd0c6cef9a72e7dc",
    "kernels.ptx": "2ff008d243a421c97c93f6c94031484d9ec165084855173386e6a491dce2d167",
}
META = (
    "outputs_sha256",
    "levels_per_mm",
    "nominal_level_per_mm",
    "shape_ha",
    "loss_gate_mm",
    "class_codes",
    "independent_pairs",
    "counts",
    "nominal_by_class",
    "parent_projection_sha256",
    "sensitive_missed_by_same_level",
)
MIB = 1 << 20


def need(ok, s):
    if not ok:
        raise ValueError(s)


def pin(s):
    need(type(s) is str and re.fullmatch("[0-9a-f]{64}", s), "external SHA-256 required")
    return s


def encode(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def equal(x, y):
    return encode(x) == encode(y)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(p, h, cap):
    pin(h)
    need(p.resolve(strict=True) == p, "alias object path")
    need(stat.S_ISREG(p.lstat().st_mode), "nonregular object")
    with p.open("rb") as f:
        size = os.fstat(f.fileno()).st_size
        need(0 <= size <= cap, "object size limit")
        b = f.read(size + 1)
    need(len(b) == size and sha(b) == h, "object bytes differ")
    return b


def obj(store, h, cap=4 * MIB):
    return read(store / pin(h)[:2] / h, h, cap)


def wall(old, new, ep, np):
    need(
        pin(ep) != pin(np) and new["format"] == old["format"] == "CFDK-edge-review-v1",
        "edge identity",
    )
    need(
        new["producer_schema"] == "CFDK-Spark-witnessed-wall-v1"
        and new["backend"] == "sealed-GB10-wall-pack-v1"
        and new["adapter_sha256"] == ADAPTER
        and new["executable_products_sha256"] == PRODUCTS,
        "GB10 wall provenance",
    )
    pp = pin(old["inputs_sha256"]["manifest.json"])
    need(
        new["parent_sha256"] == pp
        and new["parent_inputs_sha256"]
        == {k: v for k, v in old["inputs_sha256"].items() if k != "wall-pack.exe"},
        "different witness parent",
    )
    need(all(equal(new[k], old[k]) for k in META), "wall measurement differs")
    return pp


def parcel(a, b, oldpin, newpin, edge, newedge, pair):
    need(
        pin(oldpin) != pin(newpin)
        and a["edge_manifest_sha256"] == edge
        and b["edge_manifest_sha256"] == newedge
        and edge != newedge
        and equal(b["pair"], pair),
        "parcel lineage",
    )
    omit = {"backend", "qualification_sha256", "edge_manifest_sha256"}
    need(
        b["backend"]["gpu"] == "NVIDIA GB10"
        and b["qualification_sha256"] == LABEL_Q
        and equal(
            {k: v for k, v in a.items() if k not in omit},
            {k: v for k, v in b.items() if k not in omit},
        ),
        "parcel measurement or backend",
    )


def face(a, b, oldpin, newpin, ep, np, par, newpar):
    am, bm = a["measurement"], b["measurement"]
    need(
        pin(oldpin) != pin(newpin)
        and b["qualification_sha256"] == FACE_Q
        and b["device"]["gpu"] == "NVIDIA GB10"
        and b["device"]["redzones_verified"] is True
        and a["format"] == b["format"]
        and a["protocol_sha256"] == b["protocol_sha256"],
        "interface lineage",
    )
    need(
        am["edge_manifest_sha256"] == ep
        and am["parcel_manifest_sha256"] == par
        and equal(bm, dict(am, edge_manifest_sha256=np, parcel_manifest_sha256=newpar)),
        "interface parent or measurement",
    )


def index_blocks(idx, newpar, shape, mask, labels):
    h, w = shape
    need(
        idx["format"] == "CFDK-periodic-row-index-v1"
        and idx["protocol_sha256"]
        == "a05630e04c348abf03897082c74740d7ac00a4ad3246ab7094873089f58530a0"
        and idx["parcel_manifest_sha256"] == newpar
        and equal(idx["shape_ha"], shape)
        and type(idx["block_rows"]) is int
        and idx["block_rows"] == 4,
        "ROI lineage",
    )
    need(len(mask) == h * w and len(labels) == h * w * 4, "bad ROI grid")
    expected = []
    for y in range(0, h, 4):
        a, b = y * w, min(y + 4, h) * w
        expected.append([sha(mask[a:b]), sha(labels[4 * a : 4 * b])])
    need(idx["blocks"] == expected, "ROI blocks differ")


def audit(delta, store, root):
    delta = Path(delta).absolute()
    store = Path(store).absolute()
    d = json.loads(read(delta / "campaign.json", root, 4 * MIB))
    need(
        d["format"] == "CFDK-GB10-witness-lineage-v1"
        and d["frozen_campaign"] == CAMPAIGN
        and d["frozen_interfaces"] == FACES,
        "campaign protocol",
    )
    survey = json.loads(obj(store, CAMPAIGN))
    faces = json.loads(obj(store, FACES))
    plan = json.loads(obj(store, survey["recipe_sha256"]))
    entries = {e["id"]: e for e in plan["entries"]}
    need(
        set(entries) == set(d["cases"]) == set(survey["cases"]) == set(faces["cases"]),
        "campaign case set",
    )
    used = set()
    seen = set()
    maps = cells = walls = 0
    withheld = []

    def overlay(h):
        used.add(pin(h))
        return json.loads(read(delta / h, h, 4 * MIB))

    for name, entry in entries.items():
        case = d["cases"][name]
        ep = entry["manifest_sha256"]
        old = json.loads(obj(store, ep))
        base = survey["cases"][name]
        if entry["kind"] == "witness-refusal":
            need(
                base["status"] == "upstream_refused"
                and old["status"] == "datum_rejected"
                and equal(case, dict(status="withheld", reason=old["reason"], parent_sha256=ep))
                and old["reason"] == base["reason"],
                "invalid withheld record",
            )
            withheld.append(name)
            continue
        np = case["edge"]
        new = overlay(np)
        pp = wall(old, new, ep, np)
        seen.add(pp)
        parent = json.loads(obj(store, pp))
        need(
            parent["status"] in ("compared", "incomplete_coverage")
            and parent["format"] == "CFDK-remount-witness-job-v1",
            "invalid parent receipt",
        )
        need(
            case["status"] == "accepted" and case["old_edge"] == ep and case["parent"] == pp,
            "case root",
        )
        for key, v in new["parent_inputs_sha256"].items():
            if key == "manifest.json":
                continue
            expect = (
                parent["inputs_sha256"]
                if key in ("cone-fdk.exe", "kernels.ptx")
                else parent["outputs_sha256"]
            )
            need(v == expect[key], "witness member lineage")
            obj(store, v, 256 * MIB)
        for v in old["outputs_sha256"].values():
            obj(store, v, 32 * MIB)
        walls += 1
        fr = faces["cases"][name]["rows"]
        need(len(case["rows"]) == len(base["rows"]) == len(fr), "incomplete pair set")
        for r, p, f in zip(case["rows"], base["rows"], fr, strict=True):
            need(
                equal(r["pair"], p["pair"])
                and equal(r["pair"], f["pair"])
                and p["manifest_sha256"] == f["parcel_manifest_sha256"],
                "frozen pair binding",
            )
            b = overlay(r["parcel"])
            a = json.loads(obj(store, p["manifest_sha256"]))
            parcel(a, b, p["manifest_sha256"], r["parcel"], ep, np, p["pair"])
            bf = overlay(r["face"])
            face(
                json.loads(obj(store, f["result_manifest_sha256"])),
                bf,
                f["result_manifest_sha256"],
                r["face"],
                ep,
                np,
                p["manifest_sha256"],
                r["parcel"],
            )
            cert = b["certificate"]
            need(
                p["mask_sha256"] == cert["mask_u8_sha256"]
                and p["labels_sha256"] == cert["labels_i32le_sha256"]
                and f["face_sha256"] == bf["measurement"]["summary"]["face_u8_sha256"],
                "payload crosswalk",
            )
            mask = obj(store, p["mask_sha256"])
            labels = obj(store, p["labels_sha256"])
            side = obj(store, f["face_sha256"])
            need(len(side) == len(mask), "face grid")
            index_blocks(overlay(r["index"]), r["parcel"], b["shape_ha"], mask, labels)
            maps += 1
            cells += len(mask)
    need(
        {x.name for x in delta.iterdir()} == used | {"campaign.json"},
        "unaccounted overlay members",
    )
    need((walls, maps, cells, len(withheld)) == (8, 67, 988920, 2), "incomplete campaign")
    return dict(
        witness_parents=len(seen),
        wall_jobs=walls,
        maps=maps,
        cells=cells,
        withheld=sorted(withheld),
        overlay_objects=len(used),
        root_sha256=root,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("delta", type=Path)
    p.add_argument("store", type=Path)
    p.add_argument("root")
    a = p.parse_args()
    try:
        print(json.dumps(audit(a.delta, a.store, a.root), sort_keys=True))
    except (OSError, ValueError, TypeError, KeyError) as e:
        p.exit(1, f"lineage refused: {e}\n")
