"""Release acceptance against the two measured acquisitions, not a simulated case factory."""

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import review_physical_snapshot as audit  # noqa: E402

EX = ROOT / "examples/physical-peak-batch"
PIN = "95eb0678640814549ac9ba4e2b2b52b302e3bc38c759aae5186facdee95f7e31"
PARENT = "250443f4c5ddd9cb3458617a072b7f1c4d9c40b539e3bffaeb73f4d1b9ece740"
H = audit.rw.hashed


def pinned(name, sha):
    path = EX / name
    b = path.read_bytes()
    assert H(b) == sha
    return json.loads(b)


def test_actual_batch_cli_matches_external_complete_result():
    cmd = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(HERE / "review_physical_snapshot.py"),
        str(EX),
        PIN,
        "--require-complete",
    ]
    d = json.loads(subprocess.check_output(cmd))
    b = json.loads((ROOT / "reports/gb10/physical-batch-complete.json").read_bytes())
    assert d["all_cases_present"] and b["all_cases_verified"]
    assert (
        b["specimens_with_all_planned_cases"] == d["specimens_with_evidence"] == 2
        and d["committed_cases"] == b["verified_cases"] == 3
    )
    for r, o, xyz, n in zip(
        d["cases"],
        b["cases"],
        [[367, 318, 256], [87, 147, 235], [86, 100, 225]],
        [512, 256, 256],
        strict=True,
    ):
        x = r["evidence"]
        y = o["evidence"]
        assert (
            r["case"] == o["case"]
            and x["report_sha256"] == y["qualification_sha256"]
            and x["volume_aggregate"] == y["aggregate"]
            and x["first_maximum_xyz"] == xyz
            and x["plane_rows_reaggregated"] == n
            and x["peak_plane_voxels_remeasured"] == n * n
        )


def test_published_frame_and_scan_chain():
    c = pinned("catalog.json", PIN)
    p = pinned("pine-parent.json", c["pine_parent_sha256"])
    assert c["pine_parent_sha256"] == PARENT
    s = pinned("pine-scan.json", p["scan_receipt_sha256"])
    o = pinned("pine-originals.json", s["source_receipt_sha256"])
    f = pinned("pine-frames.json", p["frame_manifest_sha256"])
    assert (
        p["frame_manifest_sha256"] == s["frame_manifest_sha256"]
        and s["original_file_sha256"] == o["full_file_sha256"]
    )
    manifest = ROOT / "fixtures/pine-cone-source-v1.json"
    assert H(manifest.read_bytes()) == o["source_manifest_sha256"] == audit.PINE_SOURCE
    src = json.loads(manifest.read_bytes())
    assert len(src["files"]) == 9 and set(src["files"]) == set(o["full_file_sha256"])
    assert (
        len(f) == 721
        and sum(r["used"] is True for r in f.values()) == 720
        and f["20201118_pine_cone_0721.tif"]["used"] is False
    )
    for i in range(1, 722):
        r = f[f"20201118_pine_cone_{i:04d}.tif"]
        assert (
            r["archive"] in src["files"]
            and r["archive"].endswith(".zip")
            and audit.rw.pin(r["sha256"])
            and (r["used"] is (i <= 720))
        )
    assert s["independently_rechecked_sparse_views"] == 90
    assert all(
        p["scans"][c]["projections_sha256"] == s["scans"][c]["projections_sha256"]
        for c in ("pine90", "pine720")
    )


@pytest.mark.parametrize(
    "fault",
    [
        "pine-swap",
        "wrong-parent",
        "wrong-reference",
        "omit-dense",
        "candidate-from-sparse",
        "repeated-parent",
    ],
)
def test_actual_full_pack_rejects_mixing_densities(fault):
    c = pinned("catalog.json", PIN)
    raw = (EX / "planes.cfdk").read_bytes()
    off = 8
    parts = []
    while off < len(raw):
        n = struct.unpack_from("<I", raw, off)[0]
        parts.append(raw[off + 4 : off + 4 + n])
        off += 4 + n
    assert len(parts) == 16 and off == len(raw)
    if fault == "pine-swap":
        parts[8:12], parts[12:16] = parts[12:16], parts[8:12]
    if fault == "wrong-parent":
        c["pine_parent_sha256"] = audit.NATIVE
    if fault == "wrong-reference":
        parts[15] = parts[11]
    if fault == "omit-dense":
        del parts[12:]
    if fault == "candidate-from-sparse":
        parts[14] = parts[10]
    if fault == "repeated-parent":
        parts.insert(4, parts[3])
    raw = audit.MAGIC + b"".join(struct.pack("<I", len(p)) + p for p in parts)
    c["pack_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        audit.decode(c, raw)
