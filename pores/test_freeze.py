import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import freeze
import numpy as np
import pytest
from audit import verify_review_images
from PIL import Image


def evidence(tmp_path, monkeypatch, rejected=None, wrong=None):
    locked = {"model": "C", "files": {"kernel": "0123"}}
    monkeypatch.setattr(freeze, "identity", lambda model: locked)
    seen = []

    def replay(directory, specimen, cache):
        assert directory == tmp_path and cache == tmp_path
        seen.append(specimen)
        return freeze.NAMES["C" if specimen != wrong else "A"], {
            "volume_p95": specimen != rejected
        }

    monkeypatch.setattr(freeze, "replay", replay)
    for i in freeze.DEV:
        (tmp_path / f"sample{i}.json").write_text(
            json.dumps({"pipeline_id": freeze.token(locked)})
        )
    return locked, seen


@pytest.mark.parametrize("rejected,wrong", [(2, None), (3, None), (5, None), (None, 3)])
def test_failed_or_mixed_development_never_unblinds(tmp_path, monkeypatch, rejected, wrong):
    _, seen = evidence(tmp_path, monkeypatch, rejected, wrong)
    with pytest.raises(ValueError, match="cannot unlock"):
        freeze.certify_development(tmp_path, tmp_path, "C")
    assert set(seen).issubset(freeze.DEV)


def test_manifest_binds_all_evidence_and_build(tmp_path, monkeypatch):
    locked, seen = evidence(tmp_path, monkeypatch)
    record = freeze.certify_development(tmp_path, tmp_path, "C")
    assert seen == [2, 3, 5] and record["holdout"] == [4, 6]
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps(record))
    assert freeze.checked_manifest(manifest, tmp_path, tmp_path, "C") == freeze.token(record)
    with (tmp_path / "sample5.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="Manifest differs"):
        freeze.checked_manifest(manifest, tmp_path, tmp_path, "C")
    locked["files"]["kernel"] = "changed"
    with pytest.raises(ValueError, match="rerun"):
        freeze.checked_manifest(manifest, tmp_path, tmp_path, "C")


def test_cli_disallows_direct_holdout_access_before_io(tmp_path):
    missing = tmp_path / "must-not-be-created"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("candidate.py")),
        "--specimens",
        "4",
        "--cache",
        str(missing),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 2 and "invalid choice" in result.stderr and not missing.exists()


def test_replay_rejects_an_incorrect_review_pixel():
    raw = np.arange(12, dtype=np.uint16).reshape(2, 2, 3)
    lo, hi = np.percentile(raw, [1, 99])
    layers = []
    for plane in raw:
        image = np.clip((plane.astype(float) - lo) * 255 / (hi - lo), 0, 255).astype("u1")
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        layers.append({"image": base64.b64encode(buf.getvalue()).decode()})
    p = {"low": lo, "high": hi, "layers": layers}
    verify_review_images(p, raw)
    layers.reverse()
    with pytest.raises(AssertionError, match="Wrong raw plane"):
        verify_review_images(p, raw)
