"""Generate or recheck typed sampled-grid interfaces of each certified defect component."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from anchor_audit import pin, publish
from build_periodic_labels import sha
from interface_ledger import capacity, ledger
from periodic_certificate import require
from periodic_parcel import offline
from qualify_periodic_interfaces import BUILD_NAMES, PROTOCOL, account, native
from region_review import read_bound

FORMAT = "CFDK-certified-interface-ledger-v1"


def document(p, expected, limit=1 << 20):
    raw, _ = read_bound(p, limit, pin(expected))
    return json.loads(raw)


def reduce(bundle, face, pp, band):
    verified = offline(bundle, pp, band)
    doc = document(bundle / "manifest.json", pp)
    proof = ledger(
        bundle / "mask.u8",
        bundle / "labels.i32",
        face,
        doc["shape_ha"],
        verified["certificate"],
        band,
    )
    return dict(
        parcel_manifest_sha256=pp,
        edge_manifest_sha256=doc["edge_manifest_sha256"],
        shape_ha=doc["shape_ha"],
        pair=doc["pair"],
        summary=proof,
    )


def generate(bundle, dest, build, qualification, pp, qp, band, budget=1):
    bundle, dest, build, qualification = (
        Path(p).resolve() for p in (bundle, dest, build, qualification)
    )
    require(
        os.name == "nt"
        and not dest.exists()
        and dest.parent.is_dir()
        and all(
            not dest.is_relative_to(s) and not s.is_relative_to(dest)
            for s in (bundle, build, qualification)
        ),
        "fresh separate Windows output required",
    )
    q = document(qualification, qp)
    products = q["build"]["products"]
    require(
        q["format"] == "CFDK-periodic-interface-qualification-v1"
        and q["protocol_sha256"] == PROTOCOL
        and q["build"]["identical_builds"] == 2
        and set(products) == BUILD_NAMES
        and all(sha(build / n) == pin(v) for n, v in products.items()),
        "unqualified interface binaries",
    )
    # Do not process a source that has failed its complete certificate, even temporarily on CUDA.
    verified = offline(bundle, pp, band)
    doc = document(bundle / "manifest.json", pp)
    h, w = doc["shape_ha"]
    capacity(verified["certificate"])
    require(type(budget) is int and 1 <= budget <= 65536, "device budget")
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".interfaces-") as tmp:
        stage = Path(tmp) / "completed"
        stage.mkdir()
        face = stage / "face.u8"
        dev = native(build, bundle / "mask.u8", face, h, w, 1, band, budget)
        proof = ledger(
            bundle / "mask.u8", bundle / "labels.i32", face, [h, w], verified["certificate"], band
        )
        measured = dict(
            parcel_manifest_sha256=pp,
            edge_manifest_sha256=doc["edge_manifest_sha256"],
            shape_ha=[h, w],
            pair=doc["pair"],
            summary=proof,
        )
        d = dict(
            format=FORMAT,
            protocol_sha256=PROTOCOL,
            qualification_sha256=qp,
            device=dev,
            products=products,
            measurement=measured,
        )
        publish(stage / "manifest.json", d)
        require((stage / "manifest.json").stat().st_size <= 64 << 20, "ledger report size")
        os.rename(stage, dest)
    return d


def verify(bundle, result, pp, rp, band):
    bundle, result = Path(bundle), Path(result)
    # A maximally fragmented plane has many components. Cap the supported JSON report explicitly.
    d = document(result / "manifest.json", rp, 64 << 20)
    require(
        d["format"] == FORMAT
        and d["protocol_sha256"] == PROTOCOL
        and d["measurement"]["parcel_manifest_sha256"] == pp,
        "different interface parent",
    )
    pin(d["qualification_sha256"])
    require(
        set(d["products"]) == BUILD_NAMES and all(pin(v) for v in d["products"].values()),
        "interface product identities",
    )
    device = d["device"]
    h, w = document(bundle / "manifest.json", pp)["shape_ha"]
    account(device, h, w, 1, device["band_rows"], device["budget_bytes"] >> 20)
    actual = reduce(bundle, result / "face.u8", pp, band)
    require(actual == d["measurement"], "recount differs from archived interface ledger")
    return actual


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="mode", required=True)
    g = s.add_parser("generate")
    v = s.add_parser("verify")
    for a in (g, v):
        a.add_argument("bundle", type=Path)
        a.add_argument("result", type=Path)
        a.add_argument("--parcel-pin", required=True)
        a.add_argument("--band", type=int, default=7)
    g.add_argument("build", type=Path)
    g.add_argument("qualification", type=Path)
    g.add_argument("--qual-pin", required=True)
    g.add_argument("--budget", type=int, default=1)
    v.add_argument("--result-pin", required=True)
    a = p.parse_args()
    if a.mode == "generate":
        generate(
            a.bundle,
            a.result,
            a.build,
            a.qualification,
            a.parcel_pin,
            a.qual_pin,
            a.band,
            a.budget,
        )
        print(sha(a.result / "manifest.json"))
    else:
        print(
            json.dumps(verify(a.bundle, a.result, a.parcel_pin, a.result_pin, a.band), indent=2)
        )
