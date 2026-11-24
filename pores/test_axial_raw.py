import io
import json
from zipfile import ZipFile

import numpy as np
from pore_axial_raw import ROOT, arm_input, raw_pairs
from pore_radius_audit import iou_bound
from pore_slice_noise import md


def test_raw_axial_reversal_and_plane_ablation():
    rng = np.random.default_rng(523)
    source = rng.uniform(0, 255, (9, 5, 6)).astype("f4")
    fields = raw_pairs(source, 3)
    rev = raw_pairs(source[::-1], 3)
    for ch, back in zip(fields, rev, strict=True):
        np.testing.assert_array_equal(ch, back[::-1])
    for k, j in ((1, 1), (3, 3)):
        n = source[3:6]
        left = source[3 - k : 6 - k]
        right = source[3 + k : 6 + k]
        np.testing.assert_equal(fields[j], ((left + right) * 0.5 - n) / 24)
        np.testing.assert_equal(fields[j + 1], abs(left - right) / 48)
    x = rng.normal(size=(2, 10, 9, 8)).astype("f4")
    y = arm_input(x, "plane")
    np.testing.assert_equal(y[:, 0], x[:, 0])
    np.testing.assert_equal(y[:, 5:], x[:, 5:])
    assert not y[:, 1:5].any() and np.any(x[:, 1:5])
    np.testing.assert_equal(arm_input(x, "axial"), x)


def test_reference_iou_certificate_for_minimum_eligible_volume():
    for n in (64, 65, 77, 80, 101):
        for hits in range(n + 1):
            a, b = iou_bound(n, hits)
            assert (a, b) == (hits, n + max(64 - hits, 0))
            for overlap in range(hits + 1):
                for volume in (max(64, overlap), max(64, overlap) + 1, n + 13):
                    assert overlap * b <= a * (n + volume - overlap)
    assert iou_bound(77, 7) == (7, 134) and iou_bound(137, 65) == (65, 137)


def test_archived_ipol_control_and_foreground_counts():
    e = ROOT / "pores/evidence"

    def read(p):
        return json.loads((e / p).read_text())

    sealed = read("ipol-decisions.json")
    checked = read("ipol-check.json")
    fiji = read("bernsen-decisions.json")["specimens"]
    assert all(abs(a - b) < 0.001 for _, __, a, b in sealed["synthetic_pixel_audit"])
    with ZipFile(e / "ipol-masks.zip") as z:
        assert len(z.namelist()) == 3
        for k, r in sealed["specimens"].items():
            assert (
                int(max(r["fit"], key=lambda n: r["fit"][n]["dice"])) == r["sigma"]
                and r["masks"]["fixed"] == fiji[k]["masks"]["strict"]
            )
            assert not checked[k]["advance"]
            with np.load(io.BytesIO(z.read(f"sample{k}.npz")), allow_pickle=False) as f:
                for arm in ("ipol", "fixed"):
                    m = f[arm]
                    stats = checked[k][arm]
                    assert md(m) == r["masks"][arm] and int(m.sum()) == stats["fp"] + stats["tp"]
