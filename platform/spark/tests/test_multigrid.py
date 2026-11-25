"""Independent direct FDK grids are not crops of a resampled volume: test identities, placement and refusal."""

import copy, hashlib, json, math, os, struct, sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spark_multigrid as s
import qualify_grid_guards as guards
from spark_multigrid import g, q, v

GATE = "04d1cca1e0b30c4ff03fb7d3d084d9acf97b416703acca9238b9a31ace18216c"


def h(b):
    return hashlib.sha256(b).hexdigest()


def tiny():
    ds = [(3, 6.0, 2.0, 0, 0, 0, 3, 3, 3), (5, 5.0, 1.0, 1, 1, 3, 2, 3, 1)]
    p = [struct.pack("<" + str(n) + "f", *[float(i - 7) for i in range(n)]) for n in (27, 6)]
    return struct.pack("<8sI", b"CFDKMG01", 2) + b"".join(
        struct.pack("<Iff6I", *a) for a in ds
    ) + b"".join(p), p


def test_standalone_verifier_and_cartesian_frame(tmp_path):
    raw, parts = tiny()
    f = tmp_path / "scan"
    f.write_bytes(raw)
    out = tmp_path / "x.nrrd"
    r = v.verify(f, h(raw), 1, out)
    a, b = out.read_bytes().split(b"\n\n", 1)
    x = r["regions"][1]
    assert (
        b == parts[1]
        and x["sha256"] == h(b)
        and x["offset_bytes"] == 84 + len(parts[0])
        and x["first_voxel_center_mm"] == [-1.0, -1.0, 1.0]
        and b"sizes: 2 3 1" in a
        and b"space origin: (-1,-1,1)" in a
    )


@pytest.mark.parametrize(
    "issue",
    [
        "magic",
        "zero",
        "seventeen",
        "n",
        "nan",
        "inf",
        "pitch",
        "box",
        "depth",
        "truncated",
        "extra",
        "late-byte",
        "wrong-pin",
        "short-header",
        "wrong-region",
        "bool-index",
        "existing",
        "dangling",
        "traversal",
    ],
)
def test_no_unverified_export(tmp_path, issue):
    raw, _ = tiny()
    a = bytearray(raw)
    pin = None
    dst = tmp_path / "out.nrrd"
    region = 0
    if issue == "magic":
        a[0] ^= 1
    if issue == "zero":
        struct.pack_into("<I", a, 8, 0)
    if issue == "seventeen":
        struct.pack_into("<I", a, 8, 17)
    if issue == "n":
        struct.pack_into("<I", a, 12, 1)
    if issue == "nan":
        struct.pack_into("<f", a, 16, float("nan"))
    if issue == "inf":
        struct.pack_into("<f", a, 20, float("inf"))
    if issue == "pitch":
        struct.pack_into("<f", a, 20, 3.0)
    if issue == "box":
        struct.pack_into("<I", a, 24, 2)
    if issue == "depth":
        struct.pack_into("<I", a, 44, 0)
    if issue == "truncated":
        a.pop()
    if issue == "extra":
        a.extend(b"x")
    if issue == "late-byte":
        pin = h(raw)
        a[-1] ^= 1
    if issue == "wrong-pin":
        pin = "xyz"
    if issue == "short-header":
        a = a[:40]
    if issue == "wrong-region":
        region = 9
    if issue == "bool-index":
        region = False
    if issue == "existing":
        dst.write_bytes(b"original")
    if issue == "dangling":
        dst.symlink_to(tmp_path / "not-there")
    if issue == "traversal":
        dst = tmp_path / ".." / tmp_path.name / "out.nrrd"
    f = tmp_path / "pack"
    f.write_bytes(a)
    with pytest.raises((ValueError, OSError)):
        v.verify(f, pin or h(a), region, dst)
    assert not list(tmp_path.glob(".grid-nrrd-*"))
    if issue == "existing":
        assert dst.read_bytes() == b"original"
    elif issue == "dangling":
        assert dst.is_symlink() and not dst.exists()
    else:
        assert not os.path.lexists(dst)


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "format",
        "geometry-pin",
        "projection-pin",
        "bool-batch",
        "over-batch",
        "zero-cap",
        "over-cap",
        "empty",
        "seventeen",
        "box-type",
        "bool-coordinate",
        "overshoot",
        "zero-depth",
        "extra-grid-field",
        "zero-fov",
        "nan-fov",
        "inf-fov",
        "huge-grid",
        "tiny-grid",
        "oversize",
    ],
)
def test_recipe_refusals(fault):
    d = s.preset(s.strict.load(g.FIXTURE, q.PLAN))
    x = d["grids"][0]
    b = x["box_xyz"]
    if fault == "extra":
        d["file"] = "unexpected"
    if fault == "format":
        d["format"] = "raw"
    if fault == "geometry-pin":
        d["geometry_sha256"] = "z" * 64
    if fault == "projection-pin":
        d["projections_sha256"] = 1
    if fault == "bool-batch":
        d["batch"] = True
    if fault == "over-batch":
        d["batch"] = 257
    if fault == "zero-cap":
        d["cap_mib"] = 0
    if fault == "over-cap":
        d["cap_mib"] = 2049
    if fault == "empty":
        d["grids"] = []
    if fault == "seventeen":
        d["grids"] = [x] * 17
    if fault == "box-type":
        x["box_xyz"] = "0,0,0,192,192,192"
    if fault == "bool-coordinate":
        b[1] = False
    if fault == "overshoot":
        b[2] = 191
    if fault == "zero-depth":
        b[-1] = 0
    if fault == "extra-grid-field":
        x["name"] = "surprise"
    if fault == "zero-fov":
        x["fov_mm"] = 0
    if fault == "nan-fov":
        x["fov_mm"] = float("nan")
    if fault == "inf-fov":
        x["fov_mm"] = float("inf")
    if fault == "huge-grid":
        x["n"] = 4097
    if fault == "tiny-grid":
        x["n"] = 1
    if fault == "oversize":
        x.update(n=2048, box_xyz=[0, 0, 0, 2048, 2048, 2048])
    with pytest.raises(ValueError):
        s.validate_recipe(d)


@pytest.mark.parametrize(
    "fault",
    [
        "header-spacing",
        "origin",
        "shape",
        "offset",
        "center",
        "region-count",
        "sha",
        "native-shape",
        "reordered",
        "same-hashes-wrong-fov",
        "refs",
        "serial-no-work",
        "pairs",
        "timer",
        "median",
        "losing-route",
    ],
)
def test_certificate_rejects_tampering(fault, monkeypatch):
    old = s.strict.load
    p = old(s.QDIR / "qualification.json", s.PROFILE)
    r = p["pack"]["regions"][1]
    a = p["samples"][1]
    native = a["native"][0]
    if fault == "header-spacing":
        r["pitch_mm"] += 0.1
    if fault == "origin":
        r["origin_xyz"][2] += 1
    if fault == "shape":
        r["shape_xyz"][0] -= 1
    if fault == "offset":
        r["offset_bytes"] += 4
    if fault == "center":
        r["first_voxel_center_mm"][2] += 0.002
    if fault == "region-count":
        p["pack"]["regions"].pop()
    if fault == "sha":
        r["sha256"] = "z" * 64
    if fault == "native-shape":
        native["regions"][0]["shape_xyz"][1] += 1
    if fault == "reordered":
        native["regions"].reverse()
    if fault == "same-hashes-wrong-fov":
        p["grids"][2][1] = 40.0
    if fault == "refs":
        p["reference_512_crop_sha256"] = "0" * 64
    if fault == "serial-no-work":
        p["samples"][0]["native"].pop()
    if fault == "pairs":
        p["samples"].pop()
    if fault == "timer":
        a["process_times_ms"][0] = float("nan")
    if fault == "median":
        p["decision"]["serial_ms"] += 1
    if fault == "losing-route":
        p["decision"]["gate"] = False
    monkeypatch.setattr(
        s.strict, "load", lambda path, pin: p if pin == s.PROFILE else old(path, pin)
    )
    with pytest.raises(ValueError):
        s.certificate()


def test_real_pack_and_pinned_delivery():
    _, _, p = s.certificate()
    m = v.verify(s.QDIR / "volume.mg", s.PACK)
    t = json.loads((s.M / "multigrid-delivery-001/delivery.json").read_bytes())
    assert m == t["pack"] == p["pack"] and t["issuer_sha256"] == s.sha(s.__file__)
    assert p["decision"]["wins"] == 6 and min(r["peak_owned_bytes"] for r in p["samples"]) > 0


def test_signed_fixtures_and_refusal_suite():
    root = s.M / "multigrid-guards-001"
    a = s.strict.load(root / "guards.json", GATE)
    assert (
        a["issuer"] == s.sha(s.__file__)
        and a["script"] == s.sha(guards.__file__)
        and a["profile"] == s.PROFILE
    )
    assert len(a["refusals"]) == len({f["fault"] for f in a["refusals"]}) == 9 and all(
        f["exit"] != 0 for f in a["refusals"]
    )
    assert [r["label"] for r in a["analytic"]] == [
        "reverse-odd",
        "reverse-reordered",
        "forward-origin",
    ]
    assert [r["tool"] for r in a["sanitizers"]] == ["memcheck", "synccheck"]
    for r in a["sanitizers"]:
        f = root / (r["tool"] + ".log")
        assert s.sha(f) == r["log_sha256"] and "ERROR SUMMARY: 0 errors" in f.read_text()
    for c in a["analytic"]:
        d = c["receipt"]
        grids = s.validate_recipe(c["request"])
        s.layout(d["pack"], grids)
        assert len(d["independent"]) == 4 and all(
            r["sha256"] == d["pack"]["regions"][i]["sha256"]
            for i, r in enumerate(d["independent"])
        )


def test_real_checked_request_input_digest_before_gpu(tmp_path, monkeypatch):
    geo, p = guards.gen.inputs(tmp_path)
    rr = dict(
        format="CFDK-multigrid-request-v1",
        geometry_sha256=s.sha(geo),
        projections_sha256="0" * 64,
        batch=8,
        cap_mib=4,
        grids=guards.GRIDS,
    )
    f = tmp_path / "request"
    f.write_text(json.dumps(rr))
    out = tmp_path / "delivery"
    monkeypatch.setattr(
        q, "execute", lambda *args: pytest.fail("CUDA dispatch before byte verification")
    )
    with pytest.raises(ValueError, match="changed acquisition"):
        s.issue(out, user=(f, s.sha(f), geo, p))
    assert not out.exists() and not list(tmp_path.glob(".multigrid-issue-*"))
