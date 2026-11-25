"""A source-bound, single-threshold-pair CUDA label parcel. Offline verification never runs CUDA."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from anchor_audit import pin, publish
from common import dll_dirs
from edge_review import ROLES, thickness
from periodic_certificate import certify, require
from region_review import grid_header, read_bound
from stripe_review import review
from wall import WALL_DTYPE

KIND = "CFDK-periodic-label-parcel-v1"
LAYOUT = "scenario selected by pair; rows axial, columns azimuth; only columns wrap; i32 little-endian: -1 unknown, 0 below, positive first row-major cell + 1"


def source(job, edge_pin, anchors, band, pair):
    # This is the existing production path: every level, wall record, envelope and class census.
    # A nominal mask or two user-supplied .wall files alone are not sufficient.
    full = review(job, edge_pin, anchors, band)
    s = full["result"]
    require(
        isinstance(pair, (tuple, list))
        and len(pair) == 2
        and all(type(v) is int and 0 <= v < len(s["levels_per_mm"]) for v in pair),
        "selected pair outside source levels",
    )
    row = next(r for r in s["scenarios"] if r["pair"] == list(pair))
    raw, _ = read_bound(job / "manifest.json", 1 << 20, pin(edge_pin))
    doc = json.loads(raw)
    header, _ = read_bound(
        job / "material.grid", 4096, pin(doc["outputs_sha256"]["material.grid"])
    )
    shape, lo, hi, _ = grid_header(header)
    require(list(shape) == s["shape_ha"], "source grid changed")
    return doc, s, row, lo, hi


def extract_mask(job, dest, doc, pair, band, lo, hi):
    h, w = doc["shape_ha"]
    names = [f"{role}-{i:02}.wall" for role, i in zip(ROLES, pair, strict=True)]
    source_digests = [hashlib.sha256(), hashlib.sha256()]
    mask_digest = hashlib.sha256()
    with (
        (job / names[0]).open("rb") as a,
        (job / names[1]).open("rb") as b,
        dest.open("xb") as out,
    ):
        for y in range(0, h, band):
            size = min(band, h - y) * w * WALL_DTYPE.itemsize
            data = [f.read(size) for f in (a, b)]
            require(all(len(v) == size for v in data), "short selected wall")
            for d, v in zip(source_digests, data, strict=True):
                d.update(v)
            ref, part = (thickness(np.frombuffer(v, WALL_DTYPE), lo, hi) for v in data)
            delta = ref - part
            m = np.where(np.isnan(delta), 2, delta >= doc["loss_gate_mm"]).astype("u1").tobytes()
            out.write(m)
            mask_digest.update(m)
        require(
            not a.read(1)
            and not b.read(1)
            and all(
                d.hexdigest() == doc["outputs_sha256"][n]
                for d, n in zip(source_digests, names, strict=True)
            ),
            "selected wall digest or length changed",
        )
        out.flush()
        os.fsync(out.fileno())
    return mask_digest.hexdigest()


def native(build, qualification, qual_pin, mask, labels, h, w, budget):
    from build_periodic_labels import sha
    from qualify_periodic_labels import PROTOCOL, account, runproc

    raw, _ = read_bound(qualification, 1 << 20, pin(qual_pin))
    q = json.loads(raw)
    require(
        q["format"] == "CFDK-periodic-label-qualification-v1"
        and q["protocol_sha256"] == PROTOCOL,
        "not a qualified label backend",
    )
    products = q["build"]["products"]
    names = ("periodic-labels.exe", "periodic-labels.ptx")
    require(
        q["build"]["identical_builds"] == 2
        and set(products) == set(names)
        and all(sha(build / n) == pin(products[n]) for n in names),
        "unqualified backend bytes",
    )
    env = dict(
        os.environ,
        PATH=os.pathsep.join(map(str, dll_dirs())) + os.pathsep + os.environ.get("PATH", ""),
    )
    d, _ = runproc(
        [
            build / names[0],
            "--ptx",
            build / names[1],
            "--input",
            mask,
            "--output",
            labels,
            "--height",
            h,
            "--width",
            w,
            "--maps",
            1,
            "--budget",
            budget,
        ],
        env,
    )
    account(d, h, w, 1, budget)
    return dict(
        products=products,
        gpu=d["gpu"],
        driver=d["driver"],
        peak_owned_bytes=d["peak_owned_bytes"],
        kernel_launches=d["kernel_launches"],
        redzones_verified=d["redzones_verified"],
    )


def generate(job, dest, edge_pin, build, qualification, qual_pin, pair, anchors, band, budget):
    job, dest, build, qualification = [
        Path(p).resolve() for p in (job, dest, build, qualification)
    ]
    require(
        os.name == "nt", "CUDA parcel publisher requires Windows no-overwrite directory rename"
    )
    require(
        not dest.exists()
        and dest.parent.is_dir()
        and all(
            not dest.is_relative_to(p) and not p.is_relative_to(dest)
            for p in (job, build, qualification)
        ),
        "output must be fresh and separate",
    )
    pin(edge_pin)
    pin(qual_pin)
    require(type(budget) is int and 1 <= budget <= 65536, "device budget")
    doc, s, row, lo, hi = source(job, edge_pin, anchors, band, pair)
    h, w = s["shape_ha"]
    require(
        ((h * w + 3) // 4 * 4 + 8 * h * w + 1536) <= budget << 20,
        "device budget cannot fit selected plane",
    )
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".periodic-") as tmp:
        stage = Path(tmp) / "parcel"
        stage.mkdir()
        m = stage / "mask.u8"
        lab = stage / "labels.i32"
        mask_pin = extract_mask(job, m, doc, pair, band, lo, hi)
        device = native(build, qualification, qual_pin, m, lab, h, w, budget)
        certificate = certify(m, lab, mask_pin, [h, w], row, band)
        result = dict(
            format=KIND,
            layout=LAYOUT,
            edge_manifest_sha256=edge_pin,
            material_grid_sha256=doc["outputs_sha256"]["material.grid"],
            qualification_sha256=qual_pin,
            shape_ha=[h, w],
            producer_band_rows=band,
            levels_per_mm=s["levels_per_mm"],
            loss_gate_mm=s["loss_gate_mm"],
            pair=list(pair),
            pair_levels_per_mm=[s["levels_per_mm"][i] for i in pair],
            anchors_row_col=anchors,
            source_scenario=row,
            backend=device,
            certificate=certificate,
        )
        publish(stage / "manifest.json", result)
        os.rename(
            stage, dest
        )  # Windows fails even for an already existing empty destination directory.
    return result


def offline(bundle, expected, band):
    from edge_review import scenarios
    from stripe_frontier import Frontier
    from stripe_review import MAX_BAND_CELLS

    bundle = Path(bundle)
    raw, _ = read_bound(bundle / "manifest.json", 1 << 20, pin(expected))
    d = json.loads(raw)
    require(d["format"] == KIND and d["layout"] == LAYOUT, "parcel format")
    for key in ("edge_manifest_sha256", "material_grid_sha256", "qualification_sha256"):
        pin(d[key])
    h, w = d["shape_ha"]
    require(
        type(h) is int
        and type(w) is int
        and 1 <= h <= 4096
        and 3 <= w <= 4096
        and type(band) is int
        and 1 <= band <= 4096
        and min(band, h) * w <= MAX_BAND_CELLS,
        "offline band or grid",
    )
    levels = d["levels_per_mm"]
    require(
        type(levels) is list
        and 2 <= len(levels) <= 16
        and all(type(v) in (int, float) for v in levels)
        and scenarios(levels, levels[0]) == levels,
        "invalid parcel levels",
    )
    source_band = d["producer_band_rows"]
    require(
        type(source_band) is int
        and 1 <= source_band <= 4096
        and min(h, source_band) * w <= MAX_BAND_CELLS,
        "producer band",
    )
    pair = d["pair"]
    row = d["source_scenario"]
    anchors = d["anchors_row_col"]
    cert = d["certificate"]
    require(
        len(pair) == 2
        and all(type(i) is int and 0 <= i < len(levels) for i in pair)
        and row["pair"] == pair
        and d["pair_levels_per_mm"] == [levels[i] for i in pair]
        and type(d["loss_gate_mm"]) in (float, int)
        and np.isfinite(d["loss_gate_mm"])
        and d["loss_gate_mm"] > 0,
        "parcel pair or loss gate",
    )
    require(
        len(anchors) == 2
        and all(
            len(p) == 2 and all(type(v) is int for v in p) and 0 <= p[0] < h and 0 <= p[1] < w
            for p in anchors
        ),
        "parcel anchors",
    )
    hit, opt = Frontier(w, anchors), Frontier(w, anchors)
    blind = 0
    points = [None, None]
    sha = hashlib.sha256()
    with (bundle / "mask.u8").open("rb") as f:
        for y in range(0, h, band):
            size = min(band, h - y) * w
            b = f.read(size)
            require(len(b) == size, "short offline mask")
            sha.update(b)
            m = np.frombuffer(b, "u1").reshape(-1, w)
            require(np.all(m <= 2), "invalid offline mask")
            hit.feed(m == 1)
            opt.feed(m != 0)
            blind += int((m == 2).sum())
            for i, (yy, x) in enumerate(anchors):
                if y <= yy < y + len(m):
                    points[i] = ("below", "flagged", "unknown")[m[yy - y, x]]
        require(
            not f.read(1) and sha.hexdigest() == pin(cert["mask_u8_sha256"]),
            "offline mask SHA-256",
        )
    r = hit.result()
    state = (
        "anchor_below_gate"
        if "below" in points
        else "observed_link"
        if r["anchors_linked"]
        else "unresolved_blind_path"
        if opt.result()["anchors_linked"]
        else "separated_on_grid"
    )
    truth = dict(
        pair=pair,
        observed_components=r["components"],
        largest_observed_cells=r["largest_cells"],
        observed_flagged_cells=r["flagged_cells"],
        unknown_cells=blind,
        anchor_cells=points,
        anchor_state=state,
    )
    require(row == truth, "independent offline census or anchors differ")
    actual = certify(
        bundle / "mask.u8", bundle / "labels.i32", sha.hexdigest(), [h, w], truth, band
    )
    require(
        set(actual) == set(cert)
        and all(cert[k] == v for k, v in actual.items() if k != "peak_band_cells")
        and type(cert["peak_band_cells"]) is int
        and cert["peak_band_cells"] == min(h, source_band) * w,
        "dense parcel certificate differs",
    )
    return dict(
        format="CFDK-verified-periodic-label-parcel-v1",
        manifest_sha256=expected,
        edge_manifest_sha256=d["edge_manifest_sha256"],
        pair=pair,
        source_scenario=truth,
        certificate=actual,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    g = sub.add_parser("generate")
    for name in ("job", "dest", "build", "qualification"):
        g.add_argument(name, type=Path)
    for name in ("edge-pin", "qual-pin"):
        g.add_argument("--" + name, required=True)
    g.add_argument("--pair", nargs=2, type=int, required=True)
    g.add_argument(
        "--anchors", nargs=4, type=int, required=True, metavar=("Y1", "X1", "Y2", "X2")
    )
    g.add_argument("--budget", type=int, default=32)
    v = sub.add_parser("verify")
    v.add_argument("bundle", type=Path)
    v.add_argument("--manifest-pin", required=True)
    for parser in (g, v):
        parser.add_argument("--band", type=int, default=8)
    args = vars(p.parse_args())
    action = args.pop("action")
    if action == "generate":
        a = args["anchors"]
        args["anchors"] = [a[:2], a[2:]]
        generate(**args)
        print(hashlib.sha256((args["dest"] / "manifest.json").read_bytes()).hexdigest())
    else:
        print(
            json.dumps(
                offline(args["bundle"], args["manifest_pin"], args["band"]),
                indent=2,
                allow_nan=False,
            )
        )
