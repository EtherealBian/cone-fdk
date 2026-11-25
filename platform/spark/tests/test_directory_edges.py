"""Directory handoff and publication controls. No GPU needed for these analytic byte fixtures."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_edge_compat as builder
import edge_job_inputs as api
from qualify_directory_edges import OVERLAY


def h(b):
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("relative", list(builder.PINS))
def test_changed_parent_source_refuses_overlay(tmp_path, monkeypatch, relative):
    for name in builder.PINS:
        dst = tmp_path / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = (builder.ROOT / name).read_bytes()
        dst.write_bytes(data + (b"# drift" if name == relative else b""))
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="frozen source changed"):
        builder.build(out, OVERLAY)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["changed", "link", "unlisted", "wrong-pin"])
def test_frozen_overlay_is_not_repaired(tmp_path, mutation):
    out = tmp_path / "overlay"
    assert builder.build(out, OVERLAY) == OVERLAY
    module = out / "edge_review.py"
    pin = OVERLAY
    if mutation == "changed":
        module.write_bytes(module.read_bytes() + b"#")
    elif mutation == "link":
        module.unlink()
        module.symlink_to(builder.ROOT / "scripts/edge_review.py")
    elif mutation == "unlisted":
        (out / "extra").touch()
    else:
        pin = "0" * 64
    before = {p.name: p.lstat().st_ino for p in out.iterdir()}
    with pytest.raises(ValueError):
        builder.build(out, pin)
    assert before == {p.name: p.lstat().st_ino for p in out.iterdir()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    runtime = tmp_path / "execution"
    origin = tmp_path / "original-products"
    dest = tmp_path / "final"
    for p in (parent, runtime, origin):
        p.mkdir()
    for k in api.TOOLS:
        (runtime / k).write_bytes(("local:" + k).encode())
    local = {k: api.digest(runtime / k) for k in api.TOOLS}
    monkeypatch.setattr(api, "TOOLS", local)
    inputs = {}
    outputs = {}
    for k in ("cone-fdk.exe", "kernels.ptx"):
        (origin / k).write_bytes(("archive:" + k).encode())
        inputs[k] = api.digest(origin / k)
    for k in api.CAP:
        p = parent / k
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(("archive:" + k).encode())
        outputs[k] = api.digest(p)
    manifest = {
        "format": "CFDK-remount-witness-job-v1",
        "status": "compared",
        "outputs_sha256": outputs,
        "inputs_sha256": inputs,
    }
    mp = parent / "manifest.json"
    mp.write_text(json.dumps(manifest))
    pp = api.digest(mp)

    def run(core, **kw):
        args = dict(products=runtime, source_products=origin, budget=1, parent_sha256=pp)
        args.update(kw)
        return api.review(core, parent, dest, [1, 2, 3], **args)

    return parent, runtime, origin, dest, run


def result(out):
    out.mkdir()
    (out / "map").write_bytes(b"expected")
    doc = {"outputs_sha256": {"map": h(b"expected")}}
    (out / "manifest.json").write_text(json.dumps(doc))
    return doc


@pytest.mark.parametrize(
    "group,key",
    [("archive", k) for k in ("cone-fdk.exe", "kernels.ptx")]
    + [("runtime", k) for k in ("cone-fdk.exe", "kernels.ptx", "wall-pack.exe")]
    + [
        ("parent", k)
        for k in (
            "manifest.json",
            "reference/shells.f32",
            "inspected/material.wall",
            "material.grid",
        )
    ],
)
def test_changed_sources_no_worker(fixture, group, key):
    parent, runtime, origin, dest, run = fixture
    path = {"archive": origin, "runtime": runtime, "parent": parent}[group] / key
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="changed acquisition"):
        run(
            lambda *a, **k: pytest.fail("worker executed"),
            packed=True,
            companion_sha256=api.TOOLS["wall-pack.exe"],
        )
    assert not dest.exists() and not list(dest.parent.glob(".sealed-edge-*"))


def test_archived_source_requires_external_pin(fixture):
    *_, dest, run = fixture
    with pytest.raises(ValueError, match="externally pinned"):
        run(lambda *a, **k: pytest.fail("worker executed"), parent_sha256=None)
    assert not dest.exists()


def test_real_child_reads_distinct_sealed_views_after_originals_change(fixture):
    parent, runtime, origin, dest, run = fixture
    fd_before = len(os.listdir("/proc/self/fd"))
    links = []

    def core(src, out, levels, *, pinned_reader, products, source_products, **kw):
        from contextlib import ExitStack

        # Replace one file in each of the three original directory trees *after* admission.
        triples = [
            (parent / "reference/shells.f32", src / "reference/shells.f32"),
            (runtime / "cone-fdk.exe", products / "cone-fdk.exe"),
            (origin / "cone-fdk.exe", source_products / "cone-fdk.exe"),
        ]
        with ExitStack() as held:
            for original, alias in triples:
                raw = original.read_bytes()
                held.enter_context(pinned_reader(alias))
                links.append(alias)
                original.rename(original.with_suffix(".previous"))
                original.write_bytes(b"new-bytes")
                code = "import errno,fcntl,sys; p=sys.argv[1]; f=open(p,'rb'); assert f.read().hex()==sys.argv[2]; assert fcntl.fcntl(f,fcntl.F_GET_SEALS)&fcntl.F_SEAL_WRITE; f.close()"
                subprocess.run(
                    [sys.executable, "-I", "-S", "-B", "-c", code, str(alias), raw.hex()],
                    check=True,
                    close_fds=True,
                )
        assert (products / "cone-fdk.exe").read_bytes() != (
            source_products / "cone-fdk.exe"
        ).read_bytes()
        d = result(out)
        with pinned_reader(out / "map"):
            with pytest.raises(PermissionError):
                (out / "map").write_bytes(b"corrupt")
        assert not (out / "map").is_symlink()
        return d

    doc = run(core)
    assert (dest / "map").read_bytes() == b"expected" and all(
        not os.path.lexists(p) for p in links
    )
    assert doc["input_lease"]["separate_source_products"]
    assert len(os.listdir("/proc/self/fd")) == fd_before


@pytest.mark.parametrize(
    "mutation",
    [
        "modified-bytes",
        "unexpected-file",
        "generated-alias",
        "occupied-dest",
        "alias-output",
        "exception",
    ],
)
def test_no_partial_publication(fixture, mutation):
    parent, _, _, dest, run = fixture
    original = (parent / "material.grid").read_bytes()

    def core(src, out, levels, *, pinned_reader, **kw):
        d = result(out)
        p = out / "map"
        if mutation == "modified-bytes":
            p.write_bytes(b"bogus")
        elif mutation == "unexpected-file":
            (out / "extra").touch()
        elif mutation == "alias-output":
            p.unlink()
            p.symlink_to(src / "material.grid")
        elif mutation == "generated-alias":
            with pinned_reader(p):
                p.unlink()
                p.write_bytes(b"substitute")
        elif mutation == "occupied-dest":
            dest.mkdir()
            (dest / "keep").write_bytes(b"original")
        else:
            raise RuntimeError("interrupted kernel")
        return d

    expected = {
        "modified-bytes": (ValueError, "staged bytes differ"),
        "unexpected-file": (ValueError, "unexpected staged member"),
        "generated-alias": (ValueError, "generated alias changed"),
        "occupied-dest": (FileExistsError, "File exists"),
        "alias-output": (FileNotFoundError, "No such file"),
        "exception": (RuntimeError, "interrupted kernel"),
    }
    exc, message = expected[mutation]
    with pytest.raises(exc, match=message):
        run(core)
    assert (parent / "material.grid").read_bytes() == original
    if mutation == "occupied-dest":
        assert [p.name for p in dest.iterdir()] == ["keep"] and (
            dest / "keep"
        ).read_bytes() == b"original"
    else:
        assert not os.path.lexists(dest)
    assert not list(dest.parent.glob(".sealed-edge-*"))


def test_direct_unanchored_result_does_not_claim_provenance(fixture):
    parent, runtime, _, dest, run = fixture
    path = parent / "manifest.json"
    doc = json.loads(path.read_bytes())
    doc["inputs_sha256"] = {k: api.digest(runtime / k) for k in ("cone-fdk.exe", "kernels.ptx")}
    path.write_text(json.dumps(doc))
    out = run(lambda src, out, *a, **kw: result(out), parent_sha256=None, source_products=None)
    lease = out["input_lease"]
    assert (
        lease["parent_anchor"] == "self-consistency-only"
        and lease["separate_source_products"] is False
    )
    assert json.loads((dest / "manifest.json").read_bytes()) == out
