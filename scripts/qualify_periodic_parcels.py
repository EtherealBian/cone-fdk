"""Reproduce source-bound dense label parcels for 67 pre-existing threshold pairs and two refusals."""

import argparse
import zipfile
from collections import Counter
from pathlib import Path

from anchor_audit import pin, publish
from build_periodic_labels import sha
from periodic_parcel import generate, offline
from qualify_periodic_labels import REFERENCES, checked, need

PINS = {
    **REFERENCES,
    "packed-edge-qualification.json": "7f4e16fd0e057af4d83dd551f4348317ef6f78977951c1d5344f105c3a214f23",
    "specimen-qualification.json": "9e3f69b2900a0c61339047ba3eb0bf8aa32c7e40133d16f8866203d94b04d722",
    "stripe-batch-qualification.json": "4109fac13ccd788fbf6770074254c495e7117cc8911cd826e6926405dac00554",
}
DEMOS = {("seam-flat-change", 1, 1): "seam", ("invalid-edge-control", 1, 0): "blind"}


def archive(root, out):
    with zipfile.ZipFile(out, "x") as z:
        for name in ("manifest.json", "mask.u8", "labels.i32"):
            item = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            item.create_system = 0
            item.external_attr = 0x20
            z.writestr(
                item,
                (root / name).read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


LEGACY = "invalid-edge-control"
MATCHED = {"material.grid", "class.u8", "min-loss.f64", "max-loss.f64"} | {
    f"{role}-{level:02}.wall" for role in ("reference", "inspected") for level in (0, 1)
}


def crosswalk(ref):
    old = ref["region-qualification.json"]["cases"][LEGACY]
    proof = ref["packed-edge-qualification.json"]["cases"][LEGACY]
    matched = proof["matched_outputs_sha256"]
    need(set(matched) == MATCHED, "packed crosswalk artifact scope")
    need(
        pin(proof["single_edge_sha256"]) == old["edge_sha256"]
        and pin(proof["single_lineage_region_sha256"]) == old["region_sha256"]
        and pin(proof["packed_edge_sha256"]) != old["edge_sha256"],
        "packed crosswalk lineage",
    )
    return dict(
        parent_sha256=pin(proof["parent_sha256"]),
        single_edge_sha256=old["edge_sha256"],
        single_lineage_region_sha256=old["region_sha256"],
        packed_edge_sha256=proof["packed_edge_sha256"],
        matched_outputs_sha256={n: pin(matched[n]) for n in sorted(MATCHED)},
    )


def bind_packed(job, src, proof):
    need(src == proof["packed_edge_sha256"], "different packed source")
    manifest = checked(job / "manifest.json", src)
    matched = proof["matched_outputs_sha256"]
    need(
        manifest["inputs_sha256"]["manifest.json"] == proof["parent_sha256"]
        and manifest["outputs_sha256"] == matched,
        "different packed parent or artifacts",
    )
    need(all(sha(job / n) == v for n, v in matched.items()), "packed artifact bytes differ")


def cases(ref, n):
    for report, key, rows, digest in [
        ("specimen-qualification.json", "edge_manifest_sha256", "rows", "labels_sha256"),
        ("bridge-result.json", "edge_sha256", "rows", "labels_i32le_sha256"),
        ("region-qualification.json", "edge_sha256", "cases", "labels_i32le_sha256"),
    ]:
        if n in ref[report]["cases"]:
            d = ref[report]["cases"][n]
            parent = crosswalk(ref)["packed_edge_sha256"] if n == LEGACY else d[key]
            return parent, {tuple(r["pair"]): r[digest] for r in d[rows]}
    raise ValueError("no independently pinned cases")


def qualify(recipe, reports, build, qualification, work):
    ref = {n: checked(reports / n, pin) for n, pin in PINS.items()}
    prior = ref["stripe-batch-qualification.json"]
    plan = checked(recipe, prior["recipe_sha256"])
    qpin = sha(qualification)
    work.mkdir(parents=True, exist_ok=False)
    done = {}
    demos = {}
    cells = maps = 0
    for e in plan["entries"]:
        name = e["id"]
        job = (recipe.parent / e["job"]).resolve()
        src = e["manifest_sha256"]
        dest = work / name
        expected = prior["entries"][name]
        need(src == expected["source_manifest_sha256"], "mixed batch source")
        if e["kind"] == "witness-refusal":
            try:
                generate(
                    job, dest, src, build, qualification, qpin, [0, 0], [[0, 0], [0, 1]], 1, 32
                )
            except ValueError as err:
                need(str(err) == "not an edge review", "wrong rejection of withheld job")
            else:
                raise ValueError("refused source received labels")
            need(
                not dest.exists() and expected["status"] == "upstream_refused",
                "refusal publication",
            )
            done[name] = dict(status="upstream_refused", reason=expected["reason"])
            continue
        parent, pairs = cases(ref, name)
        need(parent == src and len(pairs) == expected["pairs"], "independent map scope")
        if name == LEGACY:
            bind_packed(job, src, crosswalk(ref))
        dest.mkdir()
        rows = []
        counts = Counter()
        for (i, j), sig in sorted(pairs.items()):
            folder = dest / f"{i}-{j}"
            d = generate(
                job,
                folder,
                src,
                build,
                qualification,
                qpin,
                [i, j],
                e["anchors_row_col"],
                e["band_rows"],
                32,
            )
            pin = sha(folder / "manifest.json")
            v = offline(folder, pin, 3)
            r = d["source_scenario"]
            counts[r["anchor_state"]] += 1
            need(
                v["certificate"]["labels_i32le_sha256"] == sig and v["source_scenario"] == r,
                "independent dense labels or recovered scenario",
            )
            cells += v["certificate"]["verified_cells"]
            maps += 1
            rows.append(
                dict(
                    pair=[i, j],
                    manifest_sha256=pin,
                    labels_sha256=sig,
                    mask_sha256=v["certificate"]["mask_u8_sha256"],
                    anchor_state=r["anchor_state"],
                )
            )
            if (name, i, j) in DEMOS:
                demo = DEMOS[name, i, j]
                z = work / f"periodic-{demo}-demo.zip"
                archive(folder, z)
                # Consume the actual portable copy from disk, not the acquisition directory.
                imported = work / f"reopened-{demo}"
                with zipfile.ZipFile(z) as package:
                    need(
                        package.namelist() == ["manifest.json", "mask.u8", "labels.i32"],
                        "archive layout",
                    )
                    package.extractall(imported)
                check = offline(imported, pin, 1)
                need(
                    check["source_scenario"] == r
                    and check["certificate"]["labels_i32le_sha256"] == sig,
                    "portable replay",
                )
                demos[demo] = dict(
                    case=name,
                    pair=[i, j],
                    zip_sha256=sha(z),
                    manifest_sha256=pin,
                    bytes=z.stat().st_size,
                    anchor_state=r["anchor_state"],
                    unknown_cells=r["unknown_cells"],
                )
        need(dict(counts) == expected["anchor_states"], "mixed anchor dispositions")
        done[name] = dict(status=expected["status"], rows=rows)
        print(name, len(rows), flush=True)
    need(
        (maps, cells, len(done), set(demos)) == (67, 988920, 10, {"blind", "seam"}),
        "parcel campaign coverage",
    )
    publish(
        work / "result.json",
        dict(
            format="CFDK-periodic-parcel-campaign-v1",
            scope="67 separately selected global pairs from eight simulated projection-derived comparison jobs, plus two withheld jobs. Cross-checked every dense label against prior external digests. Not a physical accuracy claim.",
            qualification_sha256=qpin,
            recipe_sha256=sha(recipe),
            references=PINS,
            packed_crosswalk={LEGACY: crosswalk(ref)},
            sources={
                n: sha(Path(__file__).with_name(n))
                for n in (
                    "periodic_certificate.py",
                    "periodic_parcel.py",
                    "qualify_periodic_parcels.py",
                )
            },
            maps=maps,
            pair_cells=cells,
            cases=done,
            examples=demos,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("recipe", "reports", "build", "qualification", "work"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
