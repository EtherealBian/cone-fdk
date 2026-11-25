"""Independent timing-design counterexamples, not additional physical or performance observations."""

import hashlib
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import bench_residual as producer  # noqa: E402
import review_residual_bench as audit  # noqa: E402


def pair():
    shape = [5, 7, 5]
    pins = ["a" * 64, "b" * 64]
    qp = "c" * 64
    q = dict(
        format="CFDK-paired-volume-qualification-v1",
        build_sha256=audit.BUILD,
        protocol_sha256=audit.PROTOCOL,
        shape_xyz=shape,
        source={"shape": shape, "case": "analytic", "pins": pins},
        slices_verified_by_numpy=5,
        aggregate={"voxels": 175},
    )
    b = dict(
        format="CFDK-residual-band-benchmark-v1",
        harness_sha256=audit.HARNESS,
        build_sha256=audit.BUILD,
        qualified_pair_sha256=qp,
        source_sha256=pins,
        gpu="NVIDIA GB10",
        bands=[1, 4, 7, 16],
        warmups=4,
        timed_samples=32,
        budget_mib=64,
    )
    samples = []
    for block, idxs in enumerate((*producer.DESIGN, *reversed(producer.DESIGN))):
        for i in idxs:
            t = producer.BANDS[i]
            d = min(t, 5)
            samples.append(
                dict(
                    block=block,
                    band=t,
                    owned_bytes=d * 352 + 1536,
                    launches=(5 + d - 1) // d,
                    child_elapsed_ms=float([120, 104, 100, 102][i]),
                    wall_ms=75.0,
                    read_ms=21.0,
                    kernel_ms=3.0,
                )
            )
    summary, best, small = producer.summarize(samples)
    b.update(
        samples=samples,
        summary={str(k): v for k, v in summary.items()},
        minimum_observed_child_median_band=best,
        smallest_buffer_within_five_percent_of_minimum=small,
    )
    return q, qp, b, "d" * 64


def test_known_order_and_near_best():
    result = audit.review(*pair())
    assert result["best_band"] == 7 and result["smallest_near_best_band"] == 4
    assert (
        result["summary"]["4"]["paired_child_wins_against_band_1"] == 8
        and result["summary"]["4"]["child_elapsed_ms"]["mad"] == 0
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "group",
        "short",
        "allocation",
        "bool_band",
        "bool_count",
        "launches",
        "wrong_pair",
        "source",
        "tool",
        "device",
        "nan",
        "neg",
        "bool_time",
        "kernel",
        "wall",
        "phase",
        "mad",
        "wins",
        "best",
        "small",
        "workload",
    ],
)
def test_refuses_changed_timing_or_lineage(mutation):
    q, qp, b, bp = pair()
    samples = b["samples"]
    s = samples[0]
    if mutation == "order":
        samples[:2] = samples[1::-1]
    if mutation == "group":
        s["block"] = 1
    if mutation == "short":
        samples.pop()
    if mutation == "allocation":
        s["owned_bytes"] -= 1
    if mutation == "bool_band":
        s["band"] = True
    if mutation == "bool_count":
        b["warmups"] = True
    if mutation == "launches":
        s["launches"] += 1
    if mutation == "wrong_pair":
        qp = "e" * 64
    if mutation == "source":
        b["source_sha256"] = b["source_sha256"][::-1]
    if mutation == "tool":
        b["harness_sha256"] = qp
    if mutation == "device":
        b["gpu"] = "other"
    if mutation == "nan":
        s["kernel_ms"] = float("nan")
    if mutation == "neg":
        s["read_ms"] = -0.1
    if mutation == "bool_time":
        s["kernel_ms"] = True
    if mutation == "kernel":
        s["kernel_ms"] = 99.0
    if mutation == "wall":
        s["wall_ms"] = 130.0
    if mutation == "phase":
        b["summary"]["4"]["wall_ms"]["median"] = 1.0
    if mutation == "mad":
        b["summary"]["4"]["read_ms"]["mad"] += 1
    if mutation == "wins":
        b["summary"]["4"]["paired_child_wins_against_band_1"] -= 1
    if mutation == "best":
        b["minimum_observed_child_median_band"] = 4
    if mutation == "small":
        b["smallest_buffer_within_five_percent_of_minimum"] = 7
    if mutation == "workload":
        q["aggregate"]["voxels"] += 1
    with pytest.raises(ValueError):
        audit.review(q, qp, b, bp)


def test_bounded_and_pinned(tmp_path):
    p = tmp_path / "doc"
    p.write_bytes(b"{}")
    k = hashlib.sha256(p.read_bytes()).hexdigest()
    assert audit.authenticated(p, k, 2) == {}
    for pin, bound in [(k, 1), ("a" * 64, 2), ("x", 2)]:
        with pytest.raises(ValueError):
            audit.authenticated(p, pin, bound)
