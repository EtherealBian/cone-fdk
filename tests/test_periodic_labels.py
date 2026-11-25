import hashlib
from itertools import product

import numpy as np
import pytest
from periodic_label_cpu import cpu
from qualify_periodic_labels import account, adversarial, checked, flood, performance, stress


def test_all_small_ternary_file_results(tmp_path):
    a = np.asarray(list(product(range(3), repeat=6)), dtype="u1").reshape(-1, 2, 3)
    src, out = tmp_path / "input", tmp_path / "labels"
    src.write_bytes(a.tobytes())
    cpu(src, out, 2, 3, len(a))
    assert out.read_bytes() == b"".join(flood(x).tobytes() for x in a)


def test_seam_no_axial_wrap_no_diagonal_no_unknown_bridge():
    a = np.array([[1, 2, 1, 0, 1], [0, 0, 0, 1, 0], [1, 1, 0, 0, 0]], dtype="u1")
    assert flood(a).tolist() == [[1, -1, 3, 0, 1], [0, 0, 0, 9, 0], [11, 11, 0, 0, 0]]


@pytest.mark.parametrize("kind", ["existing", "alias", "short", "bad-last", "shape"])
def test_cpu_refusal_never_publishes(tmp_path, kind):
    src, out = tmp_path / "input", tmp_path / "result"
    raw = b"\x01" * 6
    src.write_bytes({"short": raw[:-1], "bad-last": raw[:-1] + b"\x03"}.get(kind, raw))
    if kind == "existing":
        out.write_bytes(b"old")
    if kind == "alias":
        out = src
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises((ValueError, FileExistsError)):
        cpu(src, out, 2, 2 if kind == "shape" else 3, 1)
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}


def test_large_stress_fixture_has_seam_islands_and_an_unknown_aperture():
    masks = list(stress(128, 192))
    assert len(masks) == 9
    assert all(np.all(a[:, 0] == 1) and np.all(a[:, -1] == 1) for a in masks)
    assert [int((a == 2).sum()) for a in masks] == [0, 0, 1] * 3
    assert len({hashlib.sha256(a).hexdigest() for a in masks}) >= 3


def test_trusted_document_digest_precedes_decode(tmp_path):
    f = tmp_path / "ref"
    f.write_bytes(b"xxx")
    with pytest.raises(ValueError, match="source SHA"):
        checked(f, "0" * 64)


def timing(gpu):
    return [
        dict(
            order=["cpu", "gpu"] if i % 2 else ["gpu", "cpu"],
            cpu_process_ms=100,
            gpu_process_ms=x,
            cpu_label_ms=20,
            gpu_label_event_ms=2,
        )
        for i, x in enumerate(gpu)
    ]


@pytest.mark.parametrize(
    "g,claim",
    [
        ([95] * 5 + [200], True),
        ([96] * 6, False),
        ([80] * 4 + [101] * 2, False),
        ([100] * 6, False),
    ],
)
def test_two_required_speed_gates(g, claim):
    assert performance(timing(g))["faster_on_this_case"] is claim


@pytest.mark.parametrize("kind", ["nan", "inf", "negative", "order", "missing"])
def test_refuse_bad_timing_study(kind):
    rows = timing([40] * 6)
    if kind == "order":
        rows[0]["order"].reverse()
    elif kind == "missing":
        rows.pop()
    else:
        rows[1]["gpu_process_ms"] = {"nan": float("nan"), "inf": float("inf"), "negative": -1}[
            kind
        ]
    with pytest.raises(ValueError):
        performance(rows)


def accounting():
    return dict(
        format="CFDK-periodic-cuda-labels-v1",
        maps=1,
        shape_ha=[2, 3],
        h2d_payload_bytes=6,
        d2h_payload_bytes=24,
        kernel_launches=3,
        redzones_verified=True,
        budget_bytes=1 << 20,
        peak_owned_bytes=8 + 48 + 1536,
        gpu_label_event_ms=0.2,
    )


@pytest.mark.parametrize(
    "field,bad",
    [
        ("redzones_verified", 1),
        ("peak_owned_bytes", 1584),
        ("maps", 2),
        ("budget_bytes", 2 << 20),
        ("d2h_payload_bytes", 28),
        ("kernel_launches", 2),
        ("shape_ha", [3, 2]),
        ("gpu_label_event_ms", float("nan")),
    ],
)
def test_reject_bad_gpu_accounting(field, bad):
    a = accounting()
    account(a, 2, 3, 1, 1)
    a[field] = bad
    with pytest.raises(ValueError):
        account(a, 2, 3, 1, 1)


def test_adversarial_controls_include_dense_contention_and_no_axial_wrap():
    uniform, snake, checker, rims, blank = (flood(a) for a in adversarial(17, 19))
    assert np.all(uniform == 1) and set(np.unique(snake)) == {0, 1} and not blank.any()
    assert set(np.unique(rims)) == {-1, 1, 16 * 19 + 1}
    assert checker[0, 0] == checker[0, -1] and checker[0, 0] != checker[1, 1]
