"""Global settings, topology and provenance: small test records, not projected specimens."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import region_review as rr
from compare_native import sha
from edge_review import CLASSES, MAPS, ROLES
from loss_oracle import cpu_labels
from remount_loss import write_doc
from wall import WALL_DTYPE


def make_edge(job):
    job.mkdir()
    h, w = 5, 10
    shape = (h, w)
    grid = job / "material.grid"
    grid.write_text("CFDKCYL1 10 5 17\n0 0 0\n0 0 1\n1 0 0\n1 .5 .25\n")
    change = np.zeros(shape)
    change[2, [1, 3, 5]] = 1.75  # always present
    change[2, [2, 4]] = [1.125, 0.875]  # two different bridge settings
    groups = []
    for role in ROLES:
        walls = []
        for i, step in enumerate((0.25, 0, -0.25)):
            t = np.full(shape, 3.0 + step)
            if role == "inspected":
                t -= change
                if i == 2:
                    t[1, 1] = np.nan
            x = np.zeros(shape, WALL_DTYPE)
            x["inner"], x["outer"], x["thickness"] = 2, 2 + t, t
            x["inner"][np.isnan(t)] = np.nan
            x["status"][np.isnan(t)] = 1
            x.tofile(job / f"{role}-{i:02}.wall")
            walls.append(t)
        groups.append(walls)
    delta = np.stack([a - b for a in groups[0] for b in groups[1]])
    nom = delta[4] >= 1
    low, high = delta.min(0), delta.max(0)
    code = np.where(np.isnan(low), 3, np.where(low >= 1, 2, np.where(high >= 1, 1, 0))).astype(
        "u1"
    )
    for name, val in zip(MAPS, (code, low, high), strict=True):
        val.tofile(job / name)

    def counts(values):
        return dict(zip(CLASSES, np.bincount(values, minlength=4).tolist(), strict=True))

    doc = dict(
        format="CFDK-edge-review-v1",
        levels_per_mm=[1, 2, 3],
        nominal_level_per_mm=2,
        loss_gate_mm=1,
        shape_ha=list(shape),
        independent_pairs=9,
        class_codes=list(CLASSES),
        counts=counts(code.ravel()),
        nominal_by_class={
            "flagged": counts(code[nom]),
            "below": counts(code[~nom]),
            "unknown": counts([]),
        },
        outputs_sha256={p.name: sha(p) for p in job.iterdir()},
    )
    write_doc(job / "manifest.json", doc)
    return job


def test_split_merge_and_missing_coverage(tmp_path):
    job = make_edge(tmp_path / "edge")
    result = rr.run(job, tmp_path / "out", 4, sha(job / "manifest.json"))
    a, nom, b = (result["scenarios"][idx] for idx in [2, 4, 6])
    assert nom["nominal_overlap"] == [[22, 22, 3], [26, 26, 1]]
    assert b["nominal_overlap"] == [[22, 22, 1], [22, 24, 1], [26, 26, 1]]
    assert a["nominal_overlap"] == [[22, 22, 3], [26, 22, 1]]
    assert [x["region_count"] for x in [a, nom, b]] == [1, 2, 3]
    assert a["unknown_cells"] == 1 and a["regions"][0]["adjacent_unknown"]
    assert b["unknown_cells"] == 0 and not any(r["adjacent_unknown"] for r in b["regions"])
    assert result["incomplete_pair_count"] == 3
    assert result["observed_extrema"]["region_count"]["max"]["value"] == 3
    assert result["observed_union_cells"] == 5 and result["always_flagged_cells"] == 3
    assert (tmp_path / "out/manifest.json").is_file()
    with pytest.raises(FileExistsError):
        rr.run(job, tmp_path / "out", 4)


def test_periodic_not_diagonal_or_axial_wrap_and_tied_peak():
    d = np.zeros((4, 6))
    d[0, [0, 5]] = 1
    d[3, 0] = 1
    d[1, 1] = 1
    d[3, 5] = np.nan
    s, lab = rr.scenario(d, 1, np.zeros_like(d, dtype=int), 2, 5)
    assert [x["id"] for x in s["regions"]] == [1, 8, 19]
    assert lab[0, 0] == lab[0, 5] == 1 and lab[3, 5] == -1
    seam, diag, bottom = s["regions"]
    assert seam["crosses_seam"] and not seam["adjacent_unknown"]
    assert not diag["touches_axial_end"] and bottom["adjacent_unknown"]
    assert seam["peak_cell"] == [0, 0] and seam["area_mm2"] == 4


@pytest.mark.parametrize("shape", [(1, 3), (2, 7), (8, 3), (17, 31), (40, 64)])
def test_random_floodfill_and_rotation(shape):
    rng = np.random.default_rng(12345)
    for probability in (0.03, 0.22, 0.49, 0.86, 1):
        mask = rng.random(shape) < probability
        a = np.zeros(shape, WALL_DTYPE)
        b = a.copy()
        a["thickness"] = mask.astype(float)
        expected = cpu_labels(a, b, 1)
        np.testing.assert_array_equal(rr.labels(mask), expected)
        shifted = rr.labels(np.roll(mask, 2, axis=1))

        def cell_counts(g):
            return sorted(np.unique(g[g > 0], return_counts=True)[1])

        assert cell_counts(shifted) == cell_counts(expected)


@pytest.mark.parametrize(
    "mutation",
    [
        "pin",
        "bad_wall_digest",
        "no_digest",
        "forged_count",
        "envelope",
        "nominal_table",
        "extra_bytes",
        "reserved",
        "format",
        "shape",
        "grid_frame",
        "bad_radius",
        "inside",
        "pair_cap",
        "detail_cap",
    ],
)
def test_reject_without_publishing(tmp_path, mutation, monkeypatch):
    job = make_edge(tmp_path / "edge")
    m = job / "manifest.json"
    doc = json.loads(m.read_bytes())
    pin = None
    radius = 4
    out = tmp_path / "out"
    if mutation == "pin":
        pin = "0" * 64
    elif mutation == "bad_wall_digest":
        doc["outputs_sha256"]["reference-00.wall"] = "0" * 64
    elif mutation == "no_digest":
        doc["outputs_sha256"]["class.u8"] = None
    elif mutation == "forged_count":
        doc["counts"]["sensitive"] += 1
    elif mutation == "nominal_table":
        doc["nominal_by_class"]["flagged"]["always_flagged"] += 1
    elif mutation == "format":
        doc["format"] = "CFDK-remount-job-v1"
    elif mutation == "shape":
        doc["shape_ha"].reverse()
    elif mutation == "bad_radius":
        radius = 10
    elif mutation == "inside":
        out = job / "child"
    elif mutation == "pair_cap":
        monkeypatch.setattr(rr, "MAX_PAIR_CELLS", 449)
    elif mutation == "detail_cap":
        monkeypatch.setattr(rr, "MAX_DETAILS", 3)
    elif mutation == "grid_frame":
        p = job / "material.grid"
        p.write_text(p.read_text().replace("0 0 1", "0 1 1"))
        doc["outputs_sha256"][p.name] = sha(p)
    else:
        p = job / ("class.u8" if mutation == "envelope" else "reference-02.wall")
        raw = bytearray(p.read_bytes())
        if mutation == "extra_bytes":
            raw.append(0)
        else:
            raw[0 if mutation == "envelope" else 28] ^= 1
        p.write_bytes(raw)
        doc["outputs_sha256"][p.name] = sha(p)
    write_doc(m, doc)
    with pytest.raises(ValueError):
        rr.run(job, out, radius, pin)
    assert not out.exists()


def test_offline_public_example(tmp_path):
    base = Path(__file__).resolve().parents[1] / "reports"
    with zipfile.ZipFile(base / "edge-review-example.zip") as z:
        z.extractall(tmp_path / "public")
    pin = json.loads((base / "edge-qualification.json").read_bytes())["cases"]["noisy-change"][
        "review_manifest_sha256"
    ]
    doc = rr.run(tmp_path / "public", tmp_path / "out", 9.4, pin)
    assert [s["observed_flagged_cells"] for s in doc["scenarios"]] == [
        262,
        271,
        275,
        254,
        263,
        272,
        253,
        255,
        263,
    ]
    assert all(s["region_count"] == 1 and not s["unknown_cells"] for s in doc["scenarios"])
    assert (
        hashlib.sha256((tmp_path / "out/manifest.json").read_bytes()).digest()
        == hashlib.sha256((base / "region-noisy-change.json").read_bytes()).digest()
    )
