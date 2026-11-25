"""Qualify GB10 against original single/packed RTX bytes, region reports, and raw CPU oracle."""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import witnessed_wall as w
from region_review import analyze
from validate_edges import raw_oracle, raw_wall
from wall import WALL_DTYPE, measure

QUAL = "7f4e16fd0e057af4d83dd551f4348317ef6f78977951c1d5344f105c3a214f23"
ORACLES = {
    "region_review.py": "535a624dd8bf287e31a65f2e3b275fb85b800135dd04b7661b421e77e695fe9d",
    "validate_edges.py": "d749599db9d48d7ac1029b1f3b8b1dda0c868d4449ddad0766d8e3e69df3c3c7",
}
LEGACY = "144a9d873a6f3047637303dff2cdfe4dbe2cb1890b1d8945b1439b5d5da60883"
BAD = {
    "zero": "increasing positive finite",
    "descending": "increasing positive finite",
    "duplicate": "increasing positive finite",
    "too_many": "increasing positive finite",
    "nan": "invalid number",
    "negative": "increasing positive finite",
    "budget": "invalid budget",
    "short": "atlas size mismatch",
    "extra": "atlas size mismatch",
    "alias": "output aliases an input",
    "hardlink": "output aliases an input",
    "no_levels": "wall-pack --input",
    "nonfinite_radius": "expected finite cylinder fields",
}
KEYS = (
    "outputs_sha256",
    "shape_ha",
    "levels_per_mm",
    "nominal_level_per_mm",
    "loss_gate_mm",
    "class_codes",
    "independent_pairs",
    "counts",
    "nominal_by_class",
    "sensitive_missed_by_same_level",
    "parent_projection_sha256",
)


def obj(store, pin):
    p = w.object_path(store, pin)
    w.need(w.sha(p) == pin, "archived evidence digest")
    return p


def run_native(atlas, grid, out, levels, budget):
    args = [
        w.BUILD / "wall-pack",
        "--ptx",
        w.BUILD / "kernels.ptx",
        "--input",
        atlas,
        "--cylinder",
        grid,
        "--output",
        out,
        "--budget",
        budget,
    ]
    for v in levels:
        args += ["--level", v]
    return subprocess.run(list(map(str, args)), capture_output=True)


def negatives(tmp, key):
    atlas = tmp / "input"
    grid = tmp / "grid"
    out = tmp / "output"
    atlas.write_bytes(b"\0" * (13 * 7 * 4))
    out.write_bytes(b"keep")
    header = "CFDKCYL1 13 1 7\n0 0 0\n0 0 1\n1 0 0\n"
    grid.write_text(header + "2 .25 .1\n")
    levels = [0.2, 0.5]
    budget = 1
    variants = {
        "zero": [0],
        "descending": [0.5, 0.2],
        "duplicate": [0.2, 0.2],
        "too_many": list(range(1, 18)),
        "nan": ["nan"],
        "negative": [-1],
        "no_levels": [],
    }
    if key in variants:
        levels = variants[key]
    elif key == "budget":
        budget = 0
    elif key == "short":
        atlas.write_bytes(atlas.read_bytes()[:-1])
    elif key == "extra":
        atlas.write_bytes(atlas.read_bytes() + b"x")
    elif key == "alias":
        out = grid
    elif key == "hardlink":
        out.unlink()
        os.link(atlas, out)
    elif key == "nonfinite_radius":
        grid.write_text(header + "1e308 1e308 .1\n")
    else:
        raise ValueError("unknown negative")
    before = {p.name: p.read_bytes() for p in tmp.iterdir()}
    r = run_native(atlas, grid, out, levels, budget)
    why = r.stderr.decode().strip()
    w.need(
        r.returncode == 2 and not r.stdout and why.startswith("wall-pack: ") and BAD[key] in why,
        "wrong native refusal " + key + ": " + why,
    )
    w.need(
        {p.name: p.read_bytes() for p in tmp.iterdir()} == before,
        "failed native call changed a file",
    )
    return dict(exit=r.returncode, stdout_bytes=len(r.stdout), stderr=why)


def analytic(tmp):
    # 7 positions below/at/above .5, plus NaN/+Inf/-Inf at each radial knot.
    ternary = list(itertools.product((0.0, 0.5, 1.0), repeat=7))
    bad = []
    for i, v in itertools.product(range(7), (np.nan, np.inf, -np.inf)):
        a = [1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.0]
        a[i] = v
        bad.append(a)
    rows = 9
    matrix = np.tile(np.array(ternary + bad, "<f4"), (rows, 1)).T.copy()
    width = len(ternary) + len(bad)
    n = width * rows
    grid = tmp / "grid"
    grid.write_text(f"CFDKCYL1 {width} {rows} 7\n0 0 0\n0 0 1\n1 0 0\n2 .25 .1\n")
    atlas = tmp / "atlas"
    matrix.tofile(atlas)
    pack = tmp / "pack"
    levels = [i / 16 for i in range(1, 17)]
    r = run_native(atlas, grid, pack, levels, 1)
    w.need(r.returncode == 0, r.stderr.decode())
    meta = json.loads(r.stdout)
    w.need(meta["gpu"] == "NVIDIA GB10" and meta["tiles"] > 1, "multi-tile GB10 required")
    size = w.old.check_stats(meta, grid, atlas, pack, levels, 1)
    digests = w.old.slices(pack, size, len(levels))
    radii = 2 + np.arange(7) * 0.25
    with pack.open("rb") as f:
        for j, v in enumerate(levels):
            arr = np.fromfile(f, WALL_DTYPE, n)
            status, t = raw_wall(matrix.astype("f8"), v, radii)
            s = tmp / f"single-{j}"
            report = measure(atlas, grid, s, v, budget=1)
            w.need(
                w.sha(s) == digests[j]
                and report["redzones_verified"]
                and report["peak_owned_bytes"] == meta["peak_owned_bytes"],
                "legacy wall bytes or accounting",
            )
            np.testing.assert_array_equal(status, arr["status"])
            np.testing.assert_array_equal(t, arr["thickness"])
            w.need(not arr["reserved"].any(), "reserved bits")
            rejected = status != 0
            for field in ("inner", "outer", "thickness"):
                w.need(
                    (arr[field][rejected].view("<u8") == 0x7FF8000000000000).all(),
                    "noncanonical rejection NaN",
                )
            # Endpoints from the first and last suprathreshold samples, not CUDA transition scan.
            keep = np.flatnonzero(~rejected)
            take = matrix[:, keep].astype("f8")
            mask = take > v
            first = mask.argmax(0)
            last = 6 - mask[::-1].argmax(0)
            ii = np.arange(len(keep))
            for jj, name in ((first - 1, "inner"), (last, "outer")):
                a, b = take[jj, ii], take[jj + 1, ii]
                expected = radii[jj] + 0.25 * (v - a) / (b - a)
                np.testing.assert_array_equal(arr[name][keep], expected)
        w.need(
            not f.read(1) and all(meta["status_counts_by_level"][7]),
            "all status classes exercised",
        )
    return dict(
        ternary_profiles=len(ternary),
        nonfinite_profiles=len(bad),
        rows=rows,
        levels=levels,
        records=n * len(levels),
        pack_sha256=w.sha(pack),
        legacy_level_sha256=digests,
        native=meta,
    )


def main(store, job):
    historical = w.ROOT / "reports/packed-edge-qualification.json"
    w.need(w.sha(historical) == QUAL, "frozen qualification pin")
    for name, pin in ORACLES.items():
        w.need(w.sha(w.SCRIPTS / name) == pin, "CPU oracle source differs")
    for name, pin in {**w.NATIVE, "cone-fdk": LEGACY}.items():
        w.need(w.sha(w.BUILD / name) == pin, "native product differs")
    q = json.loads(historical.read_bytes())
    job.mkdir(mode=0o700, exist_ok=False)
    cases = {}
    refused = {}
    for name, case in q["cases"].items():
        pp = case["parent_sha256"]
        parent = json.loads(obj(store, pp).read_bytes())
        arch = json.loads(obj(store, case["packed_edge_sha256"]).read_bytes())
        single = json.loads(obj(store, case["single_edge_sha256"]).read_bytes())
        out = job / name
        doc = w.produce(store, pp, out, arch["levels_per_mm"], budget=1)
        pin = w.sha(out / "manifest.json")
        w.need(
            doc["outputs_sha256"] == case["matched_outputs_sha256"]
            and all(doc[k] == arch[k] == single[k] for k in KEYS),
            "changed CUDA measurement",
        )
        w.need(
            doc["parent_inputs_sha256"]
            == {k: v for k, v in arch["inputs_sha256"].items() if k != "wall-pack.exe"}
            and pin not in (case["packed_edge_sha256"], case["single_edge_sha256"]),
            "false provenance",
        )
        for role, x in doc["packs"].items():
            w.need(
                x["pack_sha256"] == arch["packs"][role]["pack_sha256"]
                and x["native"]["tiles"] > 1
                and x["native"]["status_counts_by_level"]
                == arch["packs"][role]["native"]["status_counts_by_level"],
                "different packed walls",
            )
        with TemporaryDirectory(dir=job) as tmp:
            view = Path(tmp)
            for key in ("material.grid", "reference/shells.f32", "inspected/shells.f32"):
                p = view / key
                p.parent.mkdir(exist_ok=True)
                p.symlink_to(obj(store, parent["outputs_sha256"][key]))
            oracle = raw_oracle(view, out, doc)
        rr = analyze(out, 9.4, pin)
        expected = json.loads(obj(store, case["packed_region_sha256"]).read_bytes())
        w.need(
            rr["edge_manifest_sha256"] == pin
            and expected["edge_manifest_sha256"] == case["packed_edge_sha256"]
            and rr == dict(expected, edge_manifest_sha256=pin),
            "changed downstream regions beyond lineage",
        )
        w.old.write_doc(job / (name + "-regions.json"), rr)
        cases[name] = dict(
            parent_sha256=pp,
            archived_edge_sha256=case["packed_edge_sha256"],
            review_sha256=pin,
            region_sha256=w.sha(job / (name + "-regions.json")),
            outputs=doc["outputs_sha256"],
            counts=doc["counts"],
            tiles={r: p["native"]["tiles"] for r, p in doc["packs"].items()},
            **oracle,
        )
    for name, record in q["refused"].items():
        dest = job / name
        r = subprocess.run(
            [
                sys.executable,
                "-B",
                str(w.HERE / "witnessed_wall.py"),
                str(store),
                record["parent_sha256"],
                str(dest),
                "--levels",
                ".018",
                ".02",
                ".022",
            ],
            capture_output=True,
        )
        w.need(
            r.returncode == 1
            and not r.stdout
            and b"not an accepted witness" in r.stderr
            and not dest.exists(),
            "wrong upstream refusal",
        )
        refused[name] = dict(
            parent_sha256=record["parent_sha256"],
            exit=r.returncode,
            stdout_bytes=len(r.stdout),
            stderr=r.stderr.decode().strip(),
        )
    with TemporaryDirectory(dir=job) as tmp:
        an = analytic(Path(tmp))
    failures = {}
    for key in BAD:
        with TemporaryDirectory(dir=job) as tmp:
            failures[key] = negatives(Path(tmp), key)
    result = dict(
        format="CFDK-GB10-witnessed-wall-qualification-v1",
        scope="Archived simulated remounts; measured CUDA replay, not physical defect validation.",
        source_qualification_sha256=QUAL,
        adapter_sha256=w.sha(w.HERE / "witnessed_wall.py"),
        qualification_code_sha256=w.sha(Path(__file__)),
        native_sha256={**w.NATIVE, "cone-fdk": LEGACY},
        independent_sources_sha256=ORACLES,
        cases=cases,
        withheld=refused,
        analytic=an,
        native_refusals=failures,
    )
    w.old.write_doc(job / "qualification.json", result)
    print(w.sha(job / "qualification.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("store", type=Path)
    p.add_argument("job", type=Path)
    main(**vars(p.parse_args()))
