"""Qualify the optional witness-guarded wall-pack producer against frozen single-level evidence."""

import argparse
import json
from pathlib import Path

from compare_native import sha
from edge_review import ROLES
from edge_review_packed import run
from region_review import analyze
from remount_loss import write_doc
from validate_edges import raw_oracle
from validate_roi import require


def read(p):
    return json.loads(p.read_bytes())


def qualify(parents, singles, trusted, work):
    work.mkdir(parents=True, exist_ok=False)
    b = read(trusted / "wall-pack-build.json")
    ep = read(trusted / "edge-qualification.json")["cases"]
    rp = read(trusted / "region-qualification.json")["cases"]
    wp = read(trusted / "body-witness.json")["cases"]
    require(b["identical"] and b["rebuilds"] >= 2, "unqualified companion")
    cases = {}
    for name, src in (
        ("repeat", "repeat"),
        ("noisy-change", "noisy-change"),
        ("invalid-edge-control", "repeat"),
    ):
        parent, single, out = parents / src, singles / name, work / name
        record = ep[name]
        pp = wp[src]["manifest_sha256"]
        require(
            pp == record["parent_manifest_sha256"]
            and sha(single / "manifest.json")
            == record["review_manifest_sha256"]
            == rp[name]["edge_sha256"],
            "untrusted comparison",
        )
        old = read(single / "manifest.json")
        require(
            all(sha(single / n) == d for n, d in old["outputs_sha256"].items()),
            "old output differs from trusted evidence",
        )
        got = run(
            parent,
            out,
            record["levels_per_mm"],
            parent_sha256=pp,
            companion_sha256=b["executable_sha256"],
        )
        newpin = sha(out / "manifest.json")
        require(
            got["inputs_sha256"]["kernels.ptx"] == b["ptx_sha256"]
            and {k: v for k, v in got["inputs_sha256"].items() if k != "wall-pack.exe"}
            == old["inputs_sha256"]
            and got["parent_projection_sha256"] == old["parent_projection_sha256"],
            "changed upstream provenance",
        )
        require(
            got["native_wall_pack_calls"] == 2
            and got["native_wall_calls"] == 0
            and got["additional_projection_passes"] == 0,
            "incorrect execution accounting",
        )
        keys = (
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
        )
        require(all(got[k] == old[k] for k in keys), "single and packed edge outputs differ")
        idx = got["levels_per_mm"].index(got["nominal_level_per_mm"])
        for r in ROLES:
            require(
                got["stages"][r][idx] == {"source": "parent"}
                and sha(out / f"{r}-{idx:02}.wall") == sha(parent / r / "material.wall"),
                "nominal was not reused",
            )
        oracle = raw_oracle(parent, out, got)
        region = analyze(out, 9.4, newpin)
        write_doc(work / f"{name}-regions.json", region)
        # Only lineage differs. Compare a separate canonical encoding, never edit the packed evidence.
        canon = work / f"{name}-single-lineage-comparison.json"
        write_doc(canon, dict(region, edge_manifest_sha256=record["review_manifest_sha256"]))
        require(sha(canon) == rp[name]["region_sha256"], "region report differs beyond lineage")
        cases[name] = dict(
            parent_sha256=pp,
            single_edge_sha256=record["review_manifest_sha256"],
            packed_edge_sha256=newpin,
            matched_outputs_sha256=got["outputs_sha256"],
            packed_region_sha256=sha(work / f"{name}-regions.json"),
            single_lineage_region_sha256=sha(canon),
            counts=got["counts"],
            native_single_calls=old["native_wall_calls"],
            native_packed_calls=got["native_wall_pack_calls"],
            packs=got["packs"],
            **oracle,
        )
    refused = {}
    for name in ("slide", "missing-reference", "noisy-change-fixture-only"):
        pin = wp[name]["manifest_sha256"]
        try:
            run(
                parents / name,
                work / name,
                [0.018, 0.02, 0.022],
                parent_sha256=pin,
                companion_sha256=b["executable_sha256"],
            )
        except ValueError as err:
            require(
                "not an accepted witness" in str(err) and not (work / name).exists(),
                "incorrect refusal",
            )
            refused[name] = dict(parent_sha256=pin, result="withheld")
        else:
            raise ValueError("inadmissible parent produced a result")
    results = dict(
        protocol="Pinned, witness-guarded alternative producer. Same native PTX and nominal walls; compare every wall/envelope byte, independently re-interpolate raw profiles, and compare the entire region report except producer lineage to trusted single-level region evidence. Synthetic walls; no physical accuracy claim.",
        qualified_build=b,
        trusted_records_sha256={
            n: sha(trusted / n)
            for n in (
                "body-witness.json",
                "edge-qualification.json",
                "region-qualification.json",
            )
        },
        cases=cases,
        refused=refused,
    )
    write_doc(work / "qualification.json", results)
    print(f"PASS: {len(cases)} packed reviews and {len(refused)} withheld parents")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("parents", "singles", "trusted", "work"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
