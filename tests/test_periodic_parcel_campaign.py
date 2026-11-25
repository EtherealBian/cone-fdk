"""The legacy packed control can inherit native label digests only through matched wall evidence."""

import json
from pathlib import Path

import pytest
from build_periodic_labels import sha
from qualify_periodic_labels import checked
from qualify_periodic_parcels import LEGACY, MATCHED, PINS, bind_packed, cases, crosswalk


def refs():
    root = Path(__file__).resolve().parents[1] / "reports"
    return {n: checked(root / n, v) for n, v in PINS.items()}


def test_legacy_scope_is_exact():
    ref = refs()
    proof = crosswalk(ref)
    source, pairs = cases(ref, LEGACY)
    assert len(pairs) == 4 and source == proof["packed_edge_sha256"]
    with pytest.raises(ValueError, match="different packed source"):
        bind_packed(None, proof["single_edge_sha256"], proof)


@pytest.mark.parametrize(
    "field", ["single_edge_sha256", "single_lineage_region_sha256", "matched_outputs_sha256"]
)
def test_lineage_substitution(field):
    ref = refs()
    row = ref["packed-edge-qualification.json"]["cases"][LEGACY]
    if field == "matched_outputs_sha256":
        row[field].pop("min-loss.f64")
    else:
        row[field] = "0" * 64
    with pytest.raises(ValueError, match="crosswalk"):
        cases(ref, LEGACY)


@pytest.mark.parametrize("defect", [None, "parent", "listing", "truncated", "absent"])
def test_bind_rechecks_physical_artifacts(tmp_path, defect):
    for n in MATCHED:
        (tmp_path / n).write_bytes(n.encode())
    matched = {n: sha(tmp_path / n) for n in MATCHED}
    parent = "2" * 64
    meta = dict(inputs_sha256={"manifest.json": parent}, outputs_sha256=matched.copy())
    if defect == "parent":
        meta["inputs_sha256"]["manifest.json"] = "3" * 64
    elif defect == "listing":
        del meta["outputs_sha256"]["class.u8"]
    elif defect == "truncated":
        (tmp_path / "reference-01.wall").write_bytes(b"")
    elif defect == "absent":
        (tmp_path / "material.grid").unlink()
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(meta))
    src = sha(m)
    proof = dict(packed_edge_sha256=src, parent_sha256=parent, matched_outputs_sha256=matched)
    if defect is None:
        bind_packed(tmp_path, src, proof)
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            bind_packed(tmp_path, src, proof)
