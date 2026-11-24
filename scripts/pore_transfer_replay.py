"""CPU-only replay from the lossless transfer packs; optional reference TIFF re-read."""

import argparse
import base64
import csv
import io
import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from pore_transfer import (
    DEV,
    ROOT,
    SHAPE,
    Z,
    boundaries,
    census,
    correct,
    digest,
    donor_rows,
    fit,
    label,
    screen,
)

sys.path.insert(0, str(ROOT / "pores"))
from audit import require  # noqa: E402
from pore_check import reference  # noqa: E402
from validate import SPEC, final_gates, objects, score, sha, stack, verified  # noqa: E402


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        require(
            z["shape_zyx"].tolist() == list(SHAPE)
            and z["origin_xyz"].tolist() == [360, 376, Z[0]],
            "Wrong bundle region",
        )
        p = z["probability"]
        require(
            p.shape == SHAPE
            and p.dtype == np.dtype("<f4")
            and np.isfinite(p).all()
            and p.min() >= 0
            and p.max() <= 1,
            "Wrong probability field",
        )
        masks = []
        for key in ("base", "arm", "reference"):
            data = z[key]
            require(
                data.ndim == 1 and data.dtype == np.uint8 and data.size * 8 == p.size,
                "Invalid packed mask",
            )
            masks.append(np.unpackbits(data).reshape(SHAPE).view(bool))
        return p, *masks


def verify_planes(path, i, base, arm, truth, raw=None):
    html = path.read_text(encoding="utf8")
    matched = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    require(matched is not None, "Missing viewer data")
    payload = json.loads(matched[1])
    require(payload["i"] == i, "Wrong viewer specimen")
    if raw is not None:
        lo, hi = np.percentile(raw, [1, 99])
        require([lo, hi] == payload["window"], "Wrong display window")
    for k, (encoded, b, a, t) in enumerate(zip(payload["layers"], base, arm, truth, strict=True)):
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            rgb = np.asarray(image)
            require(image.mode == "RGB" and rgb.shape == (256, 256, 3), "Wrong viewer frame")
            np.testing.assert_equal(rgb[:, :, 1], (t.astype("u1") + 2 * b + 4 * a) << 5)
            if raw is not None:
                gray = np.clip((raw[k].astype("f8") - lo) * 255 / max(1, hi - lo), 0, 255).astype(
                    "u1"
                )
                np.testing.assert_equal(rgb[:, :, 0], gray)


def replay(folder, cache=None, viewers=False):
    if not __debug__:
        raise ValueError("Do not run replay with -O")
    manifest = json.loads((folder / "baseline.json").read_text())
    records, features, library = {}, {}, {}
    for i in DEV:
        rec = records[i] = json.loads((folder / f"sample{i}.json").read_text())
        published = json.loads((ROOT / f"pores/evidence/candidate-e/sample{i}.json").read_text())
        seal = manifest["specimens"][str(i)]
        require(
            manifest["model"] == published["pipeline_id"]
            and rec["specimen"] == i
            and rec["sealed"] == seal
            and rec["source_xyz"] == [360, 376, Z[0]]
            and rec["shape_zyx"] == list(SHAPE)
            and manifest["shape_zyx"] == list(SHAPE)
            and manifest["z"] == list(Z)
            and manifest["dtype"] == "<f4",
            "Different population",
        )
        require(
            all(seal[k] == published["source"][k]["sha256"] for k in ("raw", "segmented")),
            "Different archives",
        )
        p, base, arm, truth = read_npz(folder / f"sample{i}-review.npz")
        require(
            digest(p) == seal["probability"] and digest(truth) == rec["reference_u8_sha256"],
            "Unsealed input",
        )
        np.testing.assert_array_equal(base, p >= 0.5)
        if cache:
            path, _ = verified(cache, i, "segmented", False)
            np.testing.assert_array_equal(truth, stack(path, i, "segmented", *Z))
        if viewers:
            raw = None
            if cache:
                src, _ = verified(cache, i, "raw", False)
                raw = stack(src, i, "raw", *Z)
            verify_planes(folder / f"sample{i}-slices.html", i, base, arm, truth, raw)
        pred, tx = label(base), label(truth)
        require(census(tx) == rec["census"], "Census mismatch")
        features[i] = boundaries(p, pred)
        for index, key in enumerate(("inner", "outer")):
            require(
                sum(len(t[index]) for t in features[i].values()) == rec["cuda_faces"][key],
                "CUDA face count mismatch",
            )
        library[i] = donor_rows(pred, tx, features[i])
        require(library[i] == rec["donor_pairs_for_other_folds"], "Donor ledger mismatch")
        for name, mask in (("base", base), ("arm", arm)):
            a = rec[name]
            table = folder / f"sample{i}-{name}-pores.csv"
            require(
                digest(mask) == a["mask_u8_sha256"] and sha(table) == a["table_sha256"],
                "Mask or component table mismatch",
            )
            lp, rows = reference(mask, 6, (SPEC[i][2],) * 3, table, a["native"])
            require(a["exact_components"] == len(rows), "Component count mismatch")
            metrics = score(mask, truth)
            with TemporaryDirectory() as tmp:
                match = Path(tmp) / "matches.csv"
                o = objects(lp, truth, rows, match)
                require(
                    match.read_bytes() == (folder / f"sample{i}-{name}-matches.csv").read_bytes(),
                    "Match table mismatch",
                )
                with match.open(newline="") as fp:
                    matches = {int(row["ref_scipy_label"]): row for row in csv.DictReader(fp)}
            v, w = tx[1:3]
            elig = np.flatnonzero(w & (v >= 64))
            require(set(a["references"]) == {str(k) for k in elig}, "Incomplete reference rows")
            lookup = {
                int(row["id"]): int(lp[tuple(int(row["seed_" + axis]) for axis in "zyx")])
                for row in rows
            }
            for k in elig:
                row = matches.get(k)
                expected = dict(
                    reference_voxels=int(v[k]),
                    pred=None if row is None else lookup[int(row["pred_id"])],
                    voxels=None if row is None else int(row["pred_voxels"]),
                    iou=None if row is None else float(row["iou"]),
                )
                require(a["references"][str(k)] == expected, "Wrong paired reference")
            require(
                a["objects"] == o
                and a["metrics"] == metrics
                and a["gates_on_research_slab"] == final_gates(metrics, o),
                "Score mismatch",
            )
    for i in DEV:
        donors = {str(j): library[j] for j in DEV if i != j}
        rec = records[i]
        counts = {k: len(rows) for k, rows in donors.items()}
        require(rec["trained_on"] == counts, "Wrong donors")
        c = fit(donors)
        np.testing.assert_allclose(rec["coefficients"], c, rtol=1e-12, atol=1e-14)
        p, _, arm, _ = read_npz(folder / f"sample{i}-review.npz")
        derived, changes = correct(p, features[i], c)
        np.testing.assert_array_equal(arm, derived)
        require(changes == rec["adjustments"], "Wrong moves")
        sc = screen(rec["base"], rec["arm"], counts)
        sc["shared_ids"] = [int(k) for k in sc["shared_ids"]]
        require(sc == rec["screen"], "Wrong screening decision")
        print(f"Specimen {i}: replay OK, {sc['decision']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument(
        "--cache", type=Path, help="Additionally reread authenticated NIST reference TIFFs"
    )
    ap.add_argument(
        "--viewers",
        action="store_true",
        help="Check all displayed masks and, with --cache, all grayscale pixels",
    )
    a = ap.parse_args()
    replay(a.directory, a.cache, a.viewers)
