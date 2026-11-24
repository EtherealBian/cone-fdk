import csv
import hashlib
import io
import json
from pathlib import Path
from statistics import quantiles
from zipfile import ZipFile

import numpy as np
import pytest

E = Path(__file__).resolve().parent / "evidence"


def doc(p):
    return json.loads((E / p).read_text(encoding="utf8"))


@pytest.mark.parametrize("i", "235")
def test_section_pack_matches_seal_and_csv_tails(i):
    folder = E / "section-area"
    lock = doc("section-area/predictions.json")
    report = doc(f"section-area/sample{i}.json")
    diff = {}
    with ZipFile(folder / "weights-masks.zip") as z:
        assert len(z.namelist()) == 8
        for k, a in report["arms"].items():
            assert hashlib.sha256(z.read(f"{k}.npz")).hexdigest() == lock["weights"][k]
            b = z.read(f"sample{i}-{k}.npy")
            mask = np.load(io.BytesIO(b))
            rec = lock["specimens"][i]["masks"][k]
            assert hashlib.sha256(b).hexdigest() == rec["file"]
            assert mask.dtype == bool and mask.shape == (96, 256, 256)
            assert (
                hashlib.sha256(mask.tobytes()).hexdigest() == rec["mask"] == a["mask_u8_sha256"]
            )
            assert (
                rec["decision_disagreements"] == 0
                and int(mask.sum()) == a["native"]["pore_voxels"]
            )
            table = folder / f"sample{i}-{k}-pores.csv"
            assert hashlib.sha256(table.read_bytes()).hexdigest() == a["table_sha256"]
            with table.open() as f:
                assert sum(int(v["voxels"]) for v in csv.DictReader(f)) == int(mask.sum())
            with (folder / f"sample{i}-{k}-matches.csv").open() as f:
                diff[k] = {
                    int(v["ref_scipy_label"]): 100
                    * abs(int(v["pred_voxels"]) - int(v["ref_voxels"]))
                    / int(v["ref_voxels"])
                    for v in csv.DictReader(f)
                }
            assert len(diff[k]) == a["objects"]["matches"]
    common = diff["control"].keys() & diff["area"].keys()
    s = report["screen"]
    assert sorted(common) == s["shared_ids"]
    assert [
        quantiles([diff[k][v] for v in common], n=100, method="inclusive")[94]
        for k in ("control", "area")
    ] == pytest.approx(s["shared_p95"])
    assert not s["advance"]


def test_archived_bernsen_decisions_and_changes():
    from pore_slice_noise import md

    locks = doc("bernsen-decisions.json")
    scores = doc("bernsen-check.json")
    prior = doc("slice-noise-predictions.json")
    with ZipFile(E / "bernsen-masks.zip") as z:
        for i in "235":
            with np.load(io.BytesIO(z.read(f"sample{i}.npz"))) as f:
                assert {k: md(f[k]) for k in f} == locks["specimens"][i]["masks"]
                assert md(f["strict"]) == prior["specimens"][i]["masks"]["fixed"]
                assert not (f["strict"] ^ f["plugin"]).any()
                change = f["printed"] ^ f["strict"]
                assert not (change & f["low"]).any()
                assert int(change.sum()) == scores[i]["printed_disagreements"]
                assert not scores[i]["advance"]
