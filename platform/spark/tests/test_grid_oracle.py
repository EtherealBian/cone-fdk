"""Geometry, full-volume errors including boundary, ownership/lifetime, independent reader and measured controls."""

import copy, hashlib, json, math, struct, sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grid_oracle as g, review_grid_oracle as v, spark_grid_audit as a, bench_astra_lease as bench
from astra_projection_lease import projection_lease

ACCEPT = "5fbdf9308c37df5cd3e24a00e037f6847422f586e9e6fbb4f5a5d8cb77af51f1"


def digest(b):
    return hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("n", range(2, 18))
def test_minimal_centered_geometry_every_interval(n):
    for i in range(n):
        for length in range(1, n + 1 - i):
            sz, m, off = g.enclosure(n, [i, 0, n - 1, length, n, 1])
            assert (m[0] + off[0], sz[1], off[1], m[2] + off[2]) == (i, n, 0, n - 1)
            assert (n - sz[0]) % 2 == 0 and off[0] >= 0 and off[0] + length <= sz[0]
            candidates = [
                k
                for k in range(1, n + 1)
                if (n - k) % 2 == 0 and (n - k) // 2 <= i and (n + k) // 2 >= i + length
            ]
            assert sz[0] == min(candidates)
            for t in (0, length - 1):
                assert (i + t - (n - 1) / 2) == (t + off[0] - (sz[0] - 1) / 2)


@pytest.mark.parametrize(
    "box",
    [
        [1, 2, 3, 0, 2, 3],
        [0, 2, -1, 2, 2, 2],
        [True, 0, 0, 1, 1, 1],
        [1, 2, 3, 2, 1, 6],
        [1, 2],
        [0, 0, 0, 3, 1, 1.0],
    ],
)
def test_invalid_spatial_bounds(box):
    with pytest.raises(ValueError):
        g.enclosure(8, box)


@pytest.mark.parametrize("ix", [0, 7, 30, 59])
def test_all_voxels_tie_coordinates_and_denominator(ix):
    ref = np.ones((3, 4, 5), dtype="<f4")
    left = ref.copy()
    left.flat[ix] += 0.25
    row = g.score(left, ref, [10, 20, 30])
    z, y, x = np.unravel_index(ix, ref.shape)
    assert (
        row["max_parent_xyz"] == [10 + x, 20 + y, 30 + z]
        and row["rel_l2"] == pytest.approx(0.25 / math.sqrt(ref.size))
        and row["signed_mean"] == pytest.approx(0.25 / ref.size)
        and not g.passes(row, g.LIMIT)
    )
    assert row["planes"][z]["sse"] == 0.0625


@pytest.mark.parametrize("fault", ["nan", "infinite", "zero", "shape"])
def test_no_false_measurement(fault):
    v0 = np.ones((2, 3, 4), dtype="<f4")
    r = v0.copy()
    if fault == "shape":
        r = r[:, :, :3]
    elif fault == "zero":
        r[:] = 0
    else:
        r[-1, -1, -1] = {"nan": np.nan, "infinite": np.inf}[fault]
    with pytest.raises(ValueError):
        g.score(v0, r)


class Data:
    def __init__(self, lie=False):
        self.lie = lie
        self.objects = {}
        self.deleted = []

    def create(self, k, p, arr):
        self.objects[19] = arr if self.lie else arr.copy()
        return 19

    def link(self, k, p, arr):
        self.objects[19] = arr.copy() if self.lie else arr
        return 19

    def get_shared(self, i):
        return self.objects[i]

    def delete(self, i):
        del self.objects[i]
        self.deleted.append(i)


@pytest.mark.parametrize("mode", ["borrow", "copy"])
@pytest.mark.parametrize("fail", ["ok", "ownership", "body", "callback"])
def test_ownership_lifetime_even_on_exception(mode, fail):
    astra = type("Fake", (), {"data3d": Data(fail == "ownership")})()
    view = np.arange(24, dtype="<f4").reshape(2, 3, 4)
    before = view.copy()
    seen = []

    def onreg(alias, count):
        seen.append((alias, count))
        if fail == "callback":
            raise RuntimeError("callback")

    def run():
        with projection_lease(astra, {}, view, mode, onreg) as i:
            a = astra.data3d.get_shared(i)
            assert (
                a.shape == (3, 2, 4)
                and np.array_equal(a, view.transpose(1, 0, 2))
                and not np.shares_memory(a, view)
            )
            a.fill(-5)
            if fail == "body":
                raise RuntimeError("body")

    if fail == "ok":
        run()
    else:
        with pytest.raises((ValueError, RuntimeError)):
            run()
    assert (
        not astra.data3d.objects and astra.data3d.deleted == [19] and np.array_equal(before, view)
    )
    if fail != "ownership":
        assert seen == [(mode == "borrow", view.nbytes)]


@pytest.mark.parametrize("fault", ["dtype", "dimension", "mode", "empty"])
def test_invalid_lease(fault):
    source = np.ones((2, 3, 4), dtype="<f4")
    source = {"dtype": source.astype("f8"), "dimension": source[0], "empty": source[:0]}.get(
        fault, source
    )
    with pytest.raises(ValueError):
        with projection_lease(None, {}, source, "test" if fault == "mode" else "borrow"):
            pass


def test_enclosure_admission_not_output_size():
    with pytest.raises(ValueError, match="oracle envelope"):
        a.admission((720, 296, 280), ((4096, 40, (0, 0, 0, 1, 1, 1)),))
    assert (
        a.admission((720, 296, 280), ((4096, 40, (2047, 2047, 2047, 1, 1, 1)),))[
            "largest_centered_voxels"
        ]
        == 8
    )


def pack(arr, x=0, sz=5):
    return (
        struct.pack("<8sIIff6I", b"CFDKMG01", 1, sz, float(sz), 1.0, x, 1, 1, 3, 2, 2)
        + np.array(arr, dtype="<f4").tobytes()
    )


def test_detached_error_exact_and_tie(tmp_path):
    f = tmp_path / "a"
    w = tmp_path / "b"
    arr = np.arange(1, 13, dtype="<f4")
    b = pack(arr)
    arr[[4, 10]] += 0.5
    pa = pack(arr)
    f.write_bytes(pa)
    w.write_bytes(b)
    r = v.calc(f, w, digest(pa), digest(b))[0]
    assert (
        r["max_parent_xyz"] == [1, 2, 1]
        and r["signed_mean"] == pytest.approx(1 / 12)
        and r["rel_l2"] == pytest.approx(math.sqrt(0.5 / 650))
        and len(r["planes"]) == 2
    )


@pytest.mark.parametrize(
    "fault",
    [
        "old-pin",
        "trunc",
        "extra",
        "mismatch-grid",
        "nan",
        "blank",
        "pitch",
        "n",
        "magic",
        "count",
        "zero-length",
        "bad-sha",
    ],
)
def test_detached_refusals(tmp_path, fault):
    f = tmp_path / "a"
    w = tmp_path / "b"
    arr = np.arange(1, 13, dtype="<f4")
    b = pack(arr)
    left = b
    right = b
    pin = None
    if fault == "old-pin":
        left = b[:-1] + bytes([b[-1] ^ 1])
        pin = digest(b)
    if fault == "trunc":
        left = b[:-4]
    if fault == "extra":
        right = b + b"!"
    if fault == "mismatch-grid":
        left = pack(arr, x=1)
    if fault == "nan":
        arr[-1] = np.nan
        left = pack(arr)
    if fault == "blank":
        right = pack(np.zeros_like(arr))
    if fault in ("pitch", "n", "magic", "count", "zero-length"):
        left = bytearray(b)
        index = {"pitch": 20, "n": 12, "magic": 0, "count": 8, "zero-length": 36}[fault]
        left[index : index + 4] = bytes(4)
    if fault == "bad-sha":
        pin = "A" * 64
    f.write_bytes(left)
    w.write_bytes(right)
    with pytest.raises(ValueError):
        v.calc(f, w, pin or digest(left), digest(right))


@pytest.mark.parametrize(
    "fault",
    [
        "reference-origin",
        "fov",
        "offset",
        "count",
        "plane",
        "peak-xyz",
        "rel-gate",
        "forged-report",
        "short-pin",
    ],
)
def test_signed_but_false_metadata_rejected(tmp_path, fault):
    job = a.M / "grid-audit-use-001"
    base = job / "custom-audit/qualification.json"
    d = json.loads(base.read_bytes())
    r = d["regions"][1]["native_vs_astra"]
    t = d["oracle_pack"]["regions"][1]
    if fault == "reference-origin":
        t["first_voxel_center_mm"][0] += 0.1
    if fault == "offset":
        t["offset_bytes"] += 4
    if fault == "fov":
        d["regions"][1]["fov_mm"] += 0.1
    if fault == "count":
        r["voxels"] -= 1
    if fault == "plane":
        r["planes"][-1]["energy"] *= 1.1
    if fault == "peak-xyz":
        r["max_parent_xyz"][0] += 1
    if fault == "rel-gate":
        d["limits"]["native_rel_l2"] = 1e3
    q = tmp_path / "q"
    q.write_text(json.dumps(d))
    pin = digest(q.read_bytes())
    if fault == "forged-report":
        pin = "0" * 64
    if fault == "short-pin":
        pin = "f" * 4
    with pytest.raises(ValueError):
        v.check(q, pin, job / "custom-native/volume.mg", job / "custom-audit/astra.mg")


def test_measured_qualification_and_route():
    q, _ = a.qualification()
    assert (
        all(g.passes(r["native_vs_astra"], g.LIMIT) for r in q["regions"])
        and g.passes(q["regions"][1]["fresh_vs_prior_full_astra"], g.CONTROL)
        and all(not g.passes(r, g.LIMIT) for r in q["negative_shifts"])
    )
    assert [r["shift_one_voxel_xyz"] for r in q["negative_shifts"]] == [0, 1, 2]
    p = a.M / "grid-audit-use-001"
    e = g.json_checked(p / "acceptance.json", ACCEPT)
    for case in e["cases"].values():
        for n, k in case.items():
            assert a.sha(p / n) == k
    for case, k in e["reviews"].items():
        assert a.sha(p / (case + "-review.json")) == k
    assert len(e["negatives"]) == 5 and e["source"] == {
        n: a.sha(a.bench.SRC / n) for n in e["source"]
    }


@pytest.mark.parametrize("fault", ["order", "payer", "slow", "memory", "nan", "process"])
def test_benchmark_policy_fails_closed(fault):
    d = g.json_checked(a.M / "astra-grid-lease-001/benchmark.json", a.BENCH)
    r = copy.deepcopy(d["rows"])
    rows = [x for x in r if x["mode"] == "borrow" and x["block"] >= 0]
    if fault == "order":
        r.reverse()
    if fault == "payer":
        rows[0]["role"] = "distinct data pointer"
    if fault == "nan":
        rows[0]["total_ms"] = float("nan")
    if fault in ("order", "payer", "nan"):
        with pytest.raises(ValueError):
            bench.compare(r)
    else:
        for x in rows:
            x[{"slow": "total_ms", "memory": "rss_anon_kib", "process": "process_ms"}[fault]] = (
                1e9
            )
        assert bench.compare(r)["selected"] is False
