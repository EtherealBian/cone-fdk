"""A trusted manifest and byte-exact walls are required even for the optional fast path."""

import json
import shutil

import edge_review_packed as fast
import pytest
from common import BUILD
from compare_native import sha
from edge_review import run as legacy
from region_review import analyze
from remount_loss import write_doc
from test_edge_review import parent as parent
from validate_edges import raw_oracle


def invoke(src, job, levels=(1.5, 2.0, 2.5), **kw):
    kw.setdefault("parent_sha256", sha(src / "manifest.json"))
    kw.setdefault("companion_sha256", sha(BUILD / "wall-pack.exe"))
    return fast.run(src, job, levels, **kw)


def test_native_pair_packs_match_legacy_and_regions(parent, tmp_path):
    a, b = tmp_path / "regular", tmp_path / "packed"
    old = legacy(parent, a, [2.5, 1.5, 2], parent_sha256=sha(parent / "manifest.json"))
    new = invoke(parent, b)
    assert new["backend"] == "witness-wall-pack-v1" and new["counts"]["unknown"] == 3
    assert old["native_wall_calls"] == 4 and new["native_wall_calls"] == 0
    assert new["native_wall_pack_calls"] == 2
    assert new["additional_projection_passes"] == 0
    assert set(new["outputs_sha256"]) | {"manifest.json"} == {p.name for p in b.iterdir()}
    assert all((a / n).read_bytes() == (b / n).read_bytes() for n in old["outputs_sha256"])
    assert new["outputs_sha256"] == old["outputs_sha256"]
    for role in fast.ROLES:
        assert new["stages"][role][1] == {"source": "parent"}
        assert new["packs"][role]["native"]["levels_per_mm"] == [1.5, 2.5]
        assert (
            new["inputs_sha256"][role + "/material.wall"]
            == new["outputs_sha256"][role + "-01.wall"]
        )
    assert analyze(a, 1.5)["scenarios"] == analyze(b, 1.5)["scenarios"]
    raw_oracle(parent, b, new)
    with pytest.raises(FileExistsError):
        invoke(parent, b)


@pytest.mark.parametrize(
    "what",
    [
        "parent pin",
        "companion pin",
        "missing pin",
        "source",
        "witness",
        "fixture only",
        "copied calibration",
        "same scans",
        "old tool",
        "nominal count",
        "ptx",
        "status",
    ],
)
def test_bad_source_never_starts_device(parent, tmp_path, monkeypatch, what):
    src, out = tmp_path / "src", tmp_path / "job"
    shutil.copytree(parent, src)
    m = src / "manifest.json"
    d = json.loads(m.read_bytes())
    params = {}
    if what in ("parent pin", "companion pin", "missing pin"):
        params[
            {
                "parent pin": "parent_sha256",
                "companion pin": "companion_sha256",
                "missing pin": "companion_sha256",
            }[what]
        ] = None if what == "missing pin" else "0" * 64
    elif what == "source":
        (src / "reference/shells.f32").write_bytes(b"bad")
    elif what == "witness":
        d["scans"]["inspected"]["body_witnesses"][2]["residual_mm"] = 1.0
    elif what == "fixture only":
        d["format"] = "CFDK-remount-job-v1"
    elif what == "status":
        d["status"] = "datum_rejected"
    elif what == "nominal count":
        d["loss"]["coverage"]["unknown"] += 1
    else:
        k = {
            "copied calibration": "body-witness.json",
            "same scans": "inspected.f32",
            "old tool": "cone-fdk.exe",
            "ptx": "kernels.ptx",
        }[what]
        d["inputs_sha256"][k] = d["inputs_sha256"]["reference.f32"]
    write_doc(m, d)
    monkeypatch.setattr(
        fast, "measure_pack", lambda *a, **k: pytest.fail("CUDA called on invalid input")
    )
    with pytest.raises(ValueError):
        invoke(src, out, **params)
    assert not out.exists() and not list(tmp_path.glob(".packed-*"))


@pytest.mark.parametrize(
    "what", ["status histogram", "redzone", "second role", "nominal shifted"]
)
def test_late_failure_is_not_a_published_measurement(parent, tmp_path, monkeypatch, what):
    src, out = tmp_path / "src", tmp_path / "out"
    shutil.copytree(parent, src)
    if what == "nominal shifted":
        lp = src / "material-loss.json"
        loss = json.loads(lp.read_bytes())
        loss["runs"][-1][1] = 3  # same counts, wrong material cell
        write_doc(lp, loss)
        m = src / "manifest.json"
        doc = json.loads(m.read_bytes())
        doc["outputs_sha256"][lp.name] = sha(lp)
        write_doc(m, doc)
    original = fast.measure_pack

    def corrupt(atlas, *a, **kw):
        if what == "second role" and atlas.parent.name == "inspected":
            raise RuntimeError("interrupted second pack")
        meta = original(atlas, *a, **kw)
        if what == "status histogram":
            counts = meta["status_counts_by_level"][0]
            counts[0] -= 1
            counts[2] += 1
        elif what == "redzone":
            meta["redzones_verified"] = False
        return meta

    monkeypatch.setattr(fast, "measure_pack", corrupt)
    with pytest.raises(
        (ValueError, RuntimeError), match="status histogram|accounting|interrupted|nominal cell"
    ):
        invoke(src, out)
    assert not out.exists() and not list(tmp_path.glob(".packed-*"))
