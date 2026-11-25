"""Analytic and repinned cases; these are not additional measured objects."""

import io
import itertools
import struct
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from test_detached_campaign import example

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import qualify_threshold_atlas as oracle  # noqa: E402
import threshold_atlas as t  # noqa: E402

c = t.c
LAYOUT = "scenario selected by pair; rows axial, columns azimuth; only columns wrap; i32 little-endian: -1 unknown, 0 below, positive first row-major cell + 1"


def frame(m):
    a = np.asarray(m, dtype="u1").reshape(1, -1)
    return bytes(a.ravel()), oracle.periodic_ids(a).ravel().tolist()


def test_all_729_small_ternary_transitions():
    bank = [frame(p) for p in itertools.product(range(3), repeat=3)]
    for a, b in itertools.product(bank, repeat=2):
        assert t.transition(a, b) == oracle.edge(np.array(a[1]), np.array(b[1]))


def test_all_729_small_footprints():
    for p in itertools.product(range(3), repeat=6):
        m = np.array(p).reshape(2, 3)
        anchors = [[0, 0], [1, 2]]
        for mask in (m == 1, m != 0):
            assert t.footprint(mask.astype("u1").tobytes(), 2, 3, anchors) == oracle.foot(
                mask, anchors
            )


def test_union_path_is_not_a_single_setting():
    a = frame([1, 1, 0, 1, 0, 0])
    b = frame([1, 0, 1, 1, 0, 0])
    frames = {(0, 0): a, (0, 1): b, (1, 0): b, (1, 1): a}
    d, _ = t.chart(frames, 1, 6, 2, [[0, 0], [0, 3]], [(0, 2)])
    assert d["observed_link_settings"] == [] and d["union_link_without_single_setting"] is True
    assert not d["footprints"]["all_flagged"]["anchors_linked"]
    assert d["partition"] == dict(
        always_flagged=2,
        always_below=2,
        fully_observed_mixed=2,
        partially_observed=0,
        always_unknown=0,
    )
    assert d["points"][0]["labels_by_pair"] == [0, 3, 3, 0]


def test_blind_is_not_flagged_or_below():
    a = frame([1, 2, 0, 1, 0, 0])
    d, _ = t.chart({(0, 0): a}, 1, 6, 1, [[0, 0], [0, 3]], [])
    assert (
        d["partition"]["always_unknown"] == 1
        and not d["footprints"]["any_flagged_or_unknown"]["anchors_linked"]
    )
    b = frame([1, 2, 2, 1, 0, 0])
    d, _ = t.chart({(0, 0): b}, 1, 6, 1, [[0, 0], [0, 3]], [])
    assert (
        d["footprints"]["any_flagged_or_unknown"]["anchors_linked"]
        and not d["footprints"]["any_flagged"]["anchors_linked"]
        and not d["observed_link_settings"]
    )


def test_split_merge_and_hidden_support():
    a = frame([1, 1, 1, 1, 1, 0])
    b = frame([1, 1, 2, 1, 1, 0])
    e = t.transition(a, b)
    assert (
        e["overlaps"] == [[1, 1, 2], [1, 4, 2]]
        and e["split_source_ids"] == [1]
        and e["from_flagged_to_other"] == [[1, 2, 1]]
    )
    rev = t.transition(b, a)
    assert rev["merged_target_ids"] == [1] and rev["from_other_to_flagged"] == [[1, 2, 1]]


def pack(mutate=None):
    parts = []
    rows = []
    sr = []
    for i, j in itertools.product(range(2), repeat=2):
        p, f, x, m, dense, g = example()
        pair = [i, j]
        p.update(
            pair=pair,
            layout=LAYOUT,
            material_grid_sha256="5" * 64,
            levels_per_mm=[0.01, 0.02],
            pair_levels_per_mm=[[0.01, 0.02][v] for v in pair],
            loss_gate_mm=0.3,
        )
        p["source_scenario"]["pair"] = pair
        f["measurement"]["pair"] = pair
        p["certificate"].update(mask_u8_sha256=c.sha(m), labels_i32le_sha256=c.sha(dense))
        f["measurement"]["summary"]["face_u8_sha256"] = c.sha(g)
        x["blocks"] = [[c.sha(m), c.sha(dense)]]
        if i == j == 1 and mutate:
            mutate(p, f)
        pb = c.encoded(p)
        pin = c.sha(pb)
        x["parcel_manifest_sha256"] = f["measurement"]["parcel_manifest_sha256"] = pin
        payload = [pb, c.encoded(f), c.encoded(x), m, dense, g]
        parts.extend(payload)
        rows.append(
            dict(
                case="analytic",
                pair=pair,
                old_manifest_sha256="9" * 64,
                **dict(zip(c.SLOTS, map(c.sha, payload), strict=True)),
            )
        )
        sr.append(
            dict(
                pair=pair,
                manifest_sha256=pin,
                mask_sha256=c.sha(m),
                labels_sha256=c.sha(dense),
                anchor_state="observed_link",
            )
        )
    s = dict(
        format="CFDK-periodic-parcel-campaign-v1",
        recipe_sha256="1" * 64,
        qualification_sha256="2" * 64,
        maps=4,
        pair_cells=16,
        cases={
            "analytic": dict(status="complete_coverage", rows=sr),
            "refused": dict(status="upstream_refused", reason="missing"),
        },
    )
    sb = c.encoded(s)
    chain = dict(
        format="CFDK-GB10-source-to-interface-v1",
        source_recipe_sha256="1" * 64,
        label_qualification_sha256="2" * 64,
        interface_qualification_sha256="4" * 64,
        spark_survey_sha256=c.sha(sb),
        maps=4,
        cells=16,
        rows=rows,
        upstream_refusals=["refused"],
    )
    cb = c.encoded(chain)
    buf = bytearray(c.MAGIC)
    seen = set()
    for data in (cb, sb, *parts):
        if c.sha(data) not in seen:
            buf.extend(struct.pack("<I", len(data)) + data)
            seen.add(c.sha(data))
    return io.BytesIO(buf), c.sha(cb)


def test_complete_tiny_pack_and_point():
    f, p = pack()
    d = t.atlas(f, p, [("analytic", 0, 3)])
    case = d["cases"]["analytic"]
    assert (
        d["upstream_refused"] == {"refused": "missing"}
        and case["partition"]["always_flagged"] == 2
        and case["points"][0]["labels_by_pair"] == [1] * 4
    )


@pytest.mark.parametrize(
    "bad",
    [
        "edge",
        "material",
        "layout",
        "gate",
        "boolgate",
        "level",
        "duplicate-level",
        "pair-values",
        "anchors",
    ],
)
def test_resealed_wrong_identity_refused(bad):
    def change(p, f):
        if bad == "edge":
            p["edge_manifest_sha256"] = f["measurement"]["edge_manifest_sha256"] = "6" * 64
        elif bad == "material":
            p["material_grid_sha256"] = "6" * 64
        elif bad == "layout":
            p["layout"] = "axial wrap"
        elif bad == "gate":
            p["loss_gate_mm"] = 0.4
        elif bad == "boolgate":
            p["loss_gate_mm"] = True
        elif bad == "level":
            p["levels_per_mm"][0] = 0.009
        elif bad == "duplicate-level":
            p["levels_per_mm"][0] = 0.02
        elif bad == "pair-values":
            p["pair_levels_per_mm"][0] = 0.03
        else:
            p["anchors_row_col"].reverse()

    f, p = pack(change)
    with pytest.raises(ValueError):
        t.atlas(f, p)


@pytest.mark.parametrize(
    "points",
    [
        [("refused", 0, 0)],
        [("none", 0, 0)],
        [("analytic", 0, 0)] * 2,
        [("analytic", 0, 4)],
        [("analytic", 1, 0)],
        [("analytic", True, 0)],
    ],
)
def test_invalid_point(points):
    f, p = pack()
    with pytest.raises(ValueError):
        t.atlas(f, p, points)


def test_partial_or_unverified_dense_rejected():
    f, p = pack()
    raw = f.getvalue()
    for buf in (raw + b"!", raw[:-1], raw[:50]):
        with pytest.raises(ValueError):
            t.atlas(io.BytesIO(buf), p)


def test_reference_detects_altered_report():
    # The separate oracle must reject altered aggregate data, not just altered source hashes.
    pack = oracle.ROOT / "reports/spark-20261125/periodic-evidence.cfdk"
    full = t.review(pack, oracle.CHAIN)
    modified = deepcopy(full)
    modified["cases"]["critical-a"]["transitions"][0]["support_matrix"][0][0] += 1
    with pytest.raises(ValueError):
        oracle.compare(modified, oracle.ROOT.parent / "source-chain-gb10-001")
