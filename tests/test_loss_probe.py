"""A reviewer can verify a point without importing NumPy or initializing the CUDA driver."""

import io
import json
from copy import deepcopy
from zipfile import ZipFile

import pytest
from loss_probe import SAMPLE, from_bundle, probe, write_demo


def points():
    good = SAMPLE.pack(5, 9, 4, 0, 0)
    inner = SAMPLE.pack(5.5, 9, 3.5, 0, 0)
    bad = SAMPLE.pack(float("nan"), float("nan"), float("nan"), 3, 0)
    doc = dict(
        format="CFDK-loss-v1",
        rows=1,
        columns=3,
        axial_pitch_mm=0.2,
        threshold_mm=0.30,
        runs=[[0, 0, 1, 1], [0, 2, 1, -5]],
        regions=[
            dict(
                id=1,
                peak=dict(
                    row=0, column=0, loss_mm=0.5, inner_growth_mm=0.5, outer_recession_mm=0.0
                ),
            )
        ],
    )
    return doc, good * 3, inner + good + bad


@pytest.mark.parametrize(
    "column,expected", [(0, "flagged"), (1, "below_threshold"), (2, "unknown")]
)
def test_review_pack_probes(tmp_path, column, expected):
    doc, a, b = points()
    paths = [tmp_path / str(k) for k in range(4)]
    for p, raw in zip(
        paths,
        (b"CFDKCYL1 3 1 3\n0 0 0\n0 0 1\n1 0 0\n4 3 .2\n", a, b, json.dumps(doc).encode()),
        strict=True,
    ):
        p.write_bytes(raw)
    z = tmp_path / "one.zip"
    write_demo(z, *paths)
    before = z.read_bytes()
    write_demo(z, *paths)
    assert z.read_bytes() == before
    cell = from_bundle(z, 0, column)
    assert cell["status"] == expected and cell["is_reported_peak"] == (column == 0)
    if expected == "unknown":
        assert cell["delta"] is None and cell["inspected"]["status"] == "multiple"
    # Repack altered input with a fresh ZIP CRC but not a fresh member hash.
    with ZipFile(z) as reader, ZipFile(tmp_path / "changed.zip", "w") as writer:
        for n in reader.namelist():
            writer.writestr(n, reader.read(n) + (b"x" if n == "grid.txt" else b""))
    with pytest.raises(ValueError, match="digest mismatch"):
        from_bundle(tmp_path / "changed.zip", 0, column)


@pytest.mark.parametrize(
    "kind",
    ["classification", "truncated", "peak", "reserved", "bounds", "overlap", "missing-region"],
)
def test_reviewer_catches_inconsistent_point(kind):
    original, a, b = points()
    doc = deepcopy(original)
    if kind == "classification":
        doc["runs"] = []
    if kind == "truncated":
        a = a[:-1]
    if kind == "peak":
        doc["regions"][0]["peak"]["outer_recession_mm"] = 0.1
    if kind == "reserved":
        a = SAMPLE.pack(5, 9, 4, 0, 7) + a[32:]
    if kind == "overlap":
        doc["runs"] += doc["runs"]
    if kind == "missing-region":
        doc["regions"] = []
    with pytest.raises(ValueError):
        probe(doc, io.BytesIO(a), io.BytesIO(b), 0, 3 if kind == "bounds" else 0)
