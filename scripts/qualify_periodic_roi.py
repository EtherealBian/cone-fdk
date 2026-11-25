"""Cross-check detached ROI evidence on every archived parcel using direct NumPy crops."""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from periodic_roi import PROTOCOL, digest, encode, index, inspect, need

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "1d0e7ec3923b388e48232e8c460cf1ce7c9f24f6d637a0cffe8d87d757047980"
DEMOS = {
    ("seam-flat-change", 1, 1): ("seam", [18, 354, 3, 12]),
    ("invalid-edge-control", 1, 0): ("blind", [8, 12, 5, 21]),
}


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def oracle(m, lab, box):
    y, x, height, width = box
    yy = np.arange(y, y + height)
    xx = np.arange(x, x + width) % m.shape[1]
    sl = np.ix_(yy, xx)
    cut, ids = m[sl], lab[sl]
    values, counts = np.unique(ids[ids > 0], return_counts=True)
    return dict(
        mask_u8_sha256=digest(cut.tobytes()),
        labels_i32le_sha256=digest(ids.astype("<i4", copy=False).tobytes()),
        requested_cells=int(cut.size),
        below_cells=int((cut == 0).sum()),
        observed_cells=int((cut == 1).sum()),
        unknown_cells=int((cut == 2).sum()),
        observed_cells_by_global_id={
            str(int(k)): int(v) for k, v in zip(values, counts, strict=True)
        },
    ), sl


def check(d, m, lab, box):
    expected, sl = oracle(m, lab, box)
    need(
        all(d[k] == v for k, v in expected.items()),
        "crop bytes, counts or canonical IDs differ from direct full-map crop",
    )
    return sl


def portable(tag, box, source, idx, pp, ip, work, m, lab):
    out = work / (tag + "-receipt")
    result = inspect(source, idx, pp, ip, box, dest=out)
    check(result, m, lab, box)
    package = work / f"roi-{tag}-receipt.zip"
    with zipfile.ZipFile(package, "x") as z:
        for item in sorted(out.iterdir()):
            zi = zipfile.ZipInfo(item.name, (1980, 1, 1, 0, 0, 0))
            zi.create_system = 0
            zi.external_attr = 0x20
            z.writestr(zi, item.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    reopened = work / (tag + "-sent")
    with zipfile.ZipFile(package) as z:
        z.extractall(reopened)
    cmd = [
        sys.executable,
        "-I",
        "-S",
        str(ROOT / "scripts/periodic_roi.py"),
        "verify",
        str(reopened),
        "--parcel-pin",
        pp,
        "--index-pin",
        ip,
        "--box",
        *map(str, box),
    ]
    ran = subprocess.run(cmd, check=True, capture_output=True, cwd=reopened)
    need(
        json.loads(ran.stdout) == result,
        "stdlib-only detached verification returned different evidence",
    )
    return dict(
        zip_sha256=sha(package),
        zip_bytes=package.stat().st_size,
        complete_payload_bytes=m.size * 5,
        receipt=result,
    )


def qualify(survey, archive, work):
    survey, archive, work = (Path(x).resolve() for x in (survey, archive, work))
    need(
        sha(survey) == CAMPAIGN and sha(archive / "result.json") == CAMPAIGN,
        "not the qualified 67-parcel campaign",
    )
    need(
        sha(ROOT / "fixtures/periodic-roi-receipt-protocol-v1.json") == PROTOCOL,
        "changed acceptance rules",
    )
    p = json.loads(survey.read_bytes())
    work.mkdir(parents=True, exist_ok=False)
    comparisons = maps = cells = refused = 0
    reports = {}
    demos = {}
    for name, case in sorted(p["cases"].items()):
        if case["status"] == "upstream_refused":
            need(
                "rows" not in case and not (archive / name).exists(),
                "refused job got ROI evidence",
            )
            reports[name] = dict(status="upstream_refused", rows=[])
            refused += 1
            continue
        rows = []
        folder = work / name
        folder.mkdir()
        for row in case["rows"]:
            i, j = row["pair"]
            source = archive / name / f"{i}-{j}"
            pp = row["manifest_sha256"]
            idx = folder / f"{i}-{j}.json"
            ip = index(source, idx, pp, 4)
            need(
                sha(source / "labels.i32") == row["labels_sha256"]
                and sha(source / "mask.u8") == row["mask_sha256"],
                "different independent full-map evidence",
            )
            h, w = json.loads((source / "manifest.json").read_bytes())["shape_ha"]
            m = np.fromfile(source / "mask.u8", "u1").reshape(h, w)
            lab = np.fromfile(source / "labels.i32", "<i4").reshape(h, w)
            covers = np.zeros((h, w), "u1")
            receipts = []
            # Four disjoint rectangles partition the map; the first angular half crosses the seam.
            for y, dy in ((0, h // 2), (h // 2, h - h // 2)):
                for x, dx in ((w - 7, w // 2), ((w - 7 + w // 2) % w, w - w // 2)):
                    box = [y, x, dy, dx]
                    r = inspect(source, idx, pp, ip, box)
                    sl = check(r, m, lab, box)
                    covers[sl] += 1
                    receipts.append(digest(encode(r)))
                    cells += dy * dx
                    comparisons += 1
            need(np.all(covers == 1), "partition misses or overlaps cells")
            box = [h - 1, w - 1, 1, 1]
            r = inspect(source, idx, pp, ip, box)
            check(r, m, lab, box)
            receipts.append(digest(encode(r)))
            comparisons += 1
            rows.append(
                dict(
                    pair=[i, j], parcel_manifest_sha256=pp, index_sha256=ip, query_sha256=receipts
                )
            )
            maps += 1
            if (name, i, j) in DEMOS:
                tag, box = DEMOS[name, i, j]
                demos[tag] = portable(tag, box, source, idx, pp, ip, work, m, lab)
        reports[name] = dict(status=case["status"], rows=rows)
    need(
        (maps, cells, comparisons, refused, set(demos))
        == (67, 988920, 335, 2, {"seam", "blind"}),
        "different real campaign coverage",
    )
    result = dict(
        format="CFDK-periodic-roi-qualification-v1",
        protocol_sha256=PROTOCOL,
        parcel_campaign_sha256=CAMPAIGN,
        verifier_sha256=sha(ROOT / "scripts/periodic_roi.py"),
        qualification_script_sha256=sha(Path(__file__)),
        parcels=maps,
        partition_cells=cells,
        partition_and_point_queries=comparisons,
        upstream_refused=refused,
        cases=reports,
        portable_examples=demos,
        scope="Direct dense-array oracle for 67 simulated threshold pairs; detached proofs authenticate requested cells and source row blocks only. Byte equality is not physical accuracy.",
    )
    (work / "qualification.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(
        f"PASS: {maps} parcels, {cells} partition cells, {comparisons} crop checks, two detached standard-library replays"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("survey", "archive", "work"):
        p.add_argument(n, type=Path)
    qualify(**vars(p.parse_args()))
