import numpy as np
import pytest
from pore_axial_raw import PTX
from pore_seed_gpu import median7
from pore_small_seed import FACE, fill, recover
from scipy import ndimage as ndi


def test_exact_integer_rank_on_halo_and_gray_ties():
    if not (PTX / "seed-median.ptx").is_file():
        pytest.skip("Requires native seed PTX")
    v = np.random.default_rng(300).choice(
        np.array([0, 1, 32766, 32767, 65534, 65535], "u2"), size=(15, 19, 22)
    )
    result = median7(v)
    assert np.array_equal(result, ndi.median_filter(v, footprint=FACE)[4:-4, 4:-4, 4:-4])


def test_no_eligibility_grant_and_no_join():
    mask = np.zeros((18, 24, 28), bool)
    mask[4:6, 8:10, 6:8] = True
    mask[4:6, 8:10, 11:13] = True
    mask[15:17, 12:14, 19:21] = True
    raw = np.full(tuple(n + 8 for n in mask.shape), 1000, "u2")
    raw[4:-4, 4:-4, 4:-4][mask] = 50
    # Flat rank is below midpoint everywhere; both seeds may grow but never meet.
    med = np.full(mask.shape, 200, "u2")
    masks, rows = recover(mask, raw, med, med.copy())
    assert len(rows) == 3 and all(r["added"] > 0 for r in rows)
    for m in masks.values():
        assert ndi.label(m, FACE)[1] == 3 and not m[[0, -1]].any() and np.all(m[mask])
    raw.fill(120)
    _, rows = recover(mask, raw, med, med)
    assert all(r["added"] == 0 for r in rows)


def test_connected_geometry_tie_order_and_unreachable():
    seed = np.zeros((3, 3, 9), bool)
    seed[1, 1, 4] = True
    a = np.zeros_like(seed)
    a[1, 1, 1:8] = True
    a &= ~seed
    cost = ndi.distance_transform_edt(~seed) ** 2
    assert np.flatnonzero(fill(seed, a, cost, 1)).tolist() == [
        np.ravel_multi_index((1, 1, 3), a.shape)
    ]
    a[1, 1, 5] = False
    with pytest.raises(ValueError, match="Not enough"):
        fill(seed, a, cost, 4)


def test_archived_entrants_against_native_tables_and_intersections():
    import csv
    import json
    from collections import Counter

    from pore_axial_raw import sha
    from pore_axial_score import read_mask
    from pore_small_seed import ARMS, E

    seal = json.loads((E / "predictions.json").read_bytes())
    audit = json.loads((E / "overlap.json").read_bytes())["specimens"]
    summary = json.loads((E / "summary.json").read_bytes())
    assert not summary["advance"] and {k for k, b in summary["criteria"].items() if not b} == {
        "no_new_unmatched"
    }
    for i in (2, 3, 5):
        r = seal["specimens"][str(i)]
        p = E / f"sample{i}.json"
        a = json.loads(p.read_bytes())
        assert sha(p) == audit[str(i)]["score_sha256"]
        rows = json.loads((E / f"sample{i}-seeds.json").read_bytes())
        entered = {q["seed_id"]: q["final"] for q in rows if q["final"] >= 64}
        assert len(rows) == {2: 17, 3: 27, 5: 10}[i] and len(entered) == {2: 14, 3: 17, 5: 5}[i]
        for k in ARMS:
            m = read_mask(E, i, k, r["masks"][k])
            arm = a["arms"][k]
            tab = E / f"sample{i}-{k}-pores.csv"
            assert sha(tab) == arm["table_sha256"] and arm["native"]["pore_voxels"] == m.sum()
            with tab.open(newline="") as f:
                tr = {int(q["id"]): q for q in csv.DictReader(f)}
            with (E / f"sample{i}-{k}-matches.csv").open(newline="") as f:
                hits = {int(q["pred_id"]): int(q["ref_scipy_label"]) for q in csv.DictReader(f)}
            ov = audit[str(i)]["arms"][k]
            counts = Counter()
            gained = []
            for q in ov["rows"]:
                pid = q["prediction"]
                assert (
                    q["voxels"] == int(tr[pid]["voxels"]) == entered[q["seed"]]
                    and int(tr[pid]["faces"]) == 0
                )
                assert hits.get(pid) == (None if q["matched"] is None else int(q["matched"]))
                if pid not in hits:
                    counts[q["category"]] += 1
                else:
                    gained.append(hits[pid])
                assert q["iou"] == q["intersection"] / (
                    q["voxels"] + q["ref_voxels"] - q["intersection"]
                )
            assert (
                set(entered) == {q["seed"] for q in ov["rows"]}
                and counts == ov["unmatched_counts"]
            )
            c = a["comparisons"][k]
            assert (
                sorted(gained) == sorted(map(int, c["gained"]))
                and not c["lost"]
                and sum(counts.values()) == c["extra_unmatched"]
                and c["shared_p95"][0] == c["shared_p95"][1]
            )
