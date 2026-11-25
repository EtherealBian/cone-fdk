"""Recheck typed interfaces on all approved measurement parcels and independently tabulate by ID."""

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from anchor_audit import publish
from build_periodic_labels import sha
from periodic_interfaces import generate, verify
from qualify_periodic_interfaces import PROTOCOL
from qualify_periodic_labels import checked, need
from qualify_periodic_roi import CAMPAIGN


def oracle(m, lab):
    """Visit each measured square's four sides, not the encoded bytes or the streaming ledger."""
    cells = Counter()
    shared = Counter()
    faces = {}
    h, w = m.shape
    for yy, xx in np.argwhere(m == 1):
        y, x = int(yy), int(xx)
        key = int(lab[y, x])
        need(key > 0, "positive label required")
        cells[key] += 1
        if key not in faces:
            faces[key] = [[0] * 3 for _ in range(4)]
        for d, (r, c) in enumerate(((y - 1, x), (y, (x + 1) % w), (y + 1, x), (y, (x - 1) % w))):
            if not 0 <= r < h:
                faces[key][d][2] += 1
            elif m[r, c] == 2:
                faces[key][d][1] += 1
            elif m[r, c] == 0:
                faces[key][d][0] += 1
            elif lab[r, c] == key:
                shared[key] += 1
            else:
                raise ValueError("distinct adjacent component IDs")
    counters = []
    for k, n in sorted(cells.items()):
        need(shared[k] % 2 == 0, "unpaired internal face")
        counters.append(
            dict(id=k, area_cells=n, internal_undirected_edges=shared[k] // 2, faces=faces[k])
        )
    return counters


def archive(dest, source):
    with zipfile.ZipFile(dest, "x") as z:
        for name in ("manifest.json", "face.u8"):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 0
            info.external_attr = 0x20
            z.writestr(
                info,
                (source / name).read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def qualify(survey, parcels, build, qualification, work):
    survey, parcels, build, qualification, work = (
        Path(p).resolve() for p in (survey, parcels, build, qualification, work)
    )
    p = checked(survey, CAMPAIGN)
    need(sha(parcels / "result.json") == CAMPAIGN, "not the archived source directory")
    work.mkdir(parents=True, exist_ok=False)
    qp = sha(qualification)
    q = json.loads(qualification.read_bytes())
    rows = {}
    maps = cells = 0
    total = np.zeros((4, 3), np.int64)
    demo = None
    for name, case in p["cases"].items():
        if case["status"] == "upstream_refused":
            need(
                not (parcels / name).exists() and not (work / name).exists(),
                "refused source acquired faces",
            )
            rows[name] = dict(status=case["status"], rows=[])
            continue
        target = work / name
        target.mkdir()
        record = []
        for item in case["rows"]:
            i, j = item["pair"]
            pp = item["manifest_sha256"]
            bundle = parcels / name / f"{i}-{j}"
            dest = target / f"{i}-{j}"
            d = generate(bundle, dest, build, qualification, pp, qp, 7, 1)
            rp = sha(dest / "manifest.json")
            v = verify(bundle, dest, pp, rp, 1)
            s = v["summary"]
            h, w = v["shape_ha"]
            need(
                d["measurement"] == v
                and d["device"]["gpu"] == q["gpu"]
                and d["device"]["driver"] == q["driver"]
                and sha(bundle / "labels.i32") == item["labels_sha256"]
                and sha(bundle / "mask.u8") == item["mask_sha256"],
                "qualified source or GPU changed",
            )
            m = np.fromfile(bundle / "mask.u8", "u1").reshape(h, w)
            lab = np.fromfile(bundle / "labels.i32", "<i4").reshape(h, w)
            independent = oracle(m, lab)
            need(s["components"] == independent, "pointwise component/side counts differ")
            f = np.zeros((4, 3), np.int64)
            for c in independent:
                f += np.array(c["faces"])
            need(f.tolist() == s["faces_by_direction_and_kind"], "direction totals differ")
            total += f
            maps += 1
            cells += h * w
            record.append(
                dict(
                    pair=[i, j],
                    parcel_manifest_sha256=pp,
                    result_manifest_sha256=rp,
                    face_sha256=s["face_u8_sha256"],
                    components=len(independent),
                    observed_cells=s["observed_cells"],
                    faces=f.tolist(),
                )
            )
            if (name, i, j) == ("seam-flat-change", 1, 1):
                z = work / "interface-seam-demo.zip"
                archive(z, dest)
                opened = work / "reopened-seam"
                with zipfile.ZipFile(z) as package:
                    package.extractall(opened)
                check = verify(bundle, opened, pp, rp, 3)
                need(check == v, "offline example round trip")
                demo = dict(
                    case=name,
                    pair=[i, j],
                    parcel_manifest_sha256=pp,
                    result_manifest_sha256=rp,
                    zip_sha256=sha(z),
                    summary=s,
                )
        rows[name] = dict(status=case["status"], rows=record)
    need(
        maps == 67
        and cells == 988920
        and sum(c["status"] == "upstream_refused" for c in rows.values()) == 2
        and demo is not None,
        "interface campaign coverage",
    )
    publish(
        work / "qualification.json",
        dict(
            format="CFDK-periodic-interface-campaign-v1",
            scope="Sampled interfaces, not a physical perimeter. Sums below count sides separately in each global threshold pair; not independent acquisitions or unique defects.",
            protocol_sha256=PROTOCOL,
            device_qualification_sha256=qp,
            parcel_campaign_sha256=CAMPAIGN,
            maps=maps,
            paired_cells=cells,
            faces_by_direction_and_kind=total.tolist(),
            cases=rows,
            example=demo,
            sources={
                n: sha(Path(__file__).with_name(n))
                for n in (
                    "periodic_interfaces.py",
                    "interface_ledger.py",
                    "qualify_interface_parcels.py",
                )
            },
        ),
    )
    print(
        f"PASS: {maps} certified maps and {cells} face codes; independent per-component directional counts and portable replay"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("survey", "parcels", "build", "qualification", "work"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
