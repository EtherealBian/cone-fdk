"""Independent ray sampling checks, and negative controls that deliberately fail the scoring gate."""

import numpy as np
import pytest
from region_review import scenario
from specimen_phantom import TUBES, clip_z, interval, removal
from validate_specimens import GATE, design_fields, metrics


@pytest.mark.parametrize("name", list(TUBES))
def test_removed_chords_against_dense_cartesian_material_samples(name):
    tube = TUBES[name]
    phi, depth, z0, z1 = tube.flat
    nx, ny = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
    probes = [
        (phi, 0.0, (z0 + z1) / 2, 0.0),
        (phi, 1.4, (z0 + z1) / 2, 0.014),
        (118.0, 7.0, 0.0, 0.1),
        (phi, 0.0, 6.5, 0.08),
        (phi, 0.0, 10.0, 0.0),
    ]
    if tube.pocket:
        probes += [(tube.pocket[0], 0.0, 1.7, 0.0), (tube.pocket[0] + 90, 5.3, 2.1, -0.014)]
    positive = 0
    for degrees, offset, z, slope in probes:
        a = np.deg2rad(degrees)
        dx, dy, dz = np.cos(a), np.sin(a), np.array(slope)
        sx, sy, sz = -15 * dx - offset * dy, -15 * dy + offset * dx, z - 15 * slope
        od = interval(sx, sy, dx, dy, 0, 0, tube.outer)
        bore = interval(sx, sy, dx, dy, 0, 0, tube.inner)
        got = removal(tube, (sx, sy, sz), (dx, dy, dz), od, bore)
        step = 0.00025
        t = (np.arange(120000) + 0.5) * step
        x, y, zz = sx + t * dx, sy + t * dy, sz + t * dz
        inside = (
            (x * x + y * y < tube.outer**2) & (x * x + y * y > tube.inner**2) & (abs(zz) <= 7)
        )
        cut = (nx * x + ny * y >= tube.outer - depth) & (zz >= z0) & (zz <= z1)
        if tube.pocket:
            angle, e, r, lo, hi = tube.pocket
            angle = np.deg2rad(angle)
            cut |= (
                ((x - e * np.cos(angle)) ** 2 + (y - e * np.sin(angle)) ** 2 <= r * r)
                & (zz >= lo)
                & (zz <= hi)
            )
        assert abs(got - step * np.count_nonzero(inside & cut)) < 0.0011
        left = clip_z(od, sz, dz, -7, 7) - clip_z(bore, sz, dz, -7, 7) - got
        assert abs(left - step * np.count_nonzero(inside & ~cut)) < 0.0018
        positive += got > 0.1
    assert positive >= 2


@pytest.mark.parametrize("name", list(TUBES))
def test_design_peaks_and_seam_from_cad_values(name):
    tube = TUBES[name]
    fields = design_fields(tube, True)
    flat = fields["OD flat"]
    assert abs(flat.max() - tube.flat[1]) < 1e-12
    assert (flat[:, int(tube.flat[0])] == flat.max()).any()
    assert np.any((flat[:, 0] >= GATE) & (flat[:, -1] >= GATE)) == (name == "seam-flat")
    if tube.pocket:
        _, e, r, _, _ = tube.pocket
        assert abs(fields["ID pocket"].max() - (e + r - tube.inner)) < 1e-12
        assert not ((fields["ID pocket"] > 0) & (flat > 0)).any()
    assert not design_fields(tube, False)


@pytest.mark.parametrize("fault", ["none", "moved", "peak", "extra", "unknown", "seam"])
def test_scoring_rejects_different_geometry_without_realigning(fault):
    fields = design_fields(TUBES["dual-band"], True)
    delta = sum(fields.values())
    nominal = np.zeros_like(delta, dtype=int)
    if fault == "moved":
        delta = np.roll(delta, 12, axis=1)
    elif fault == "peak":
        delta.flat[delta.argmax()] += 0.13
    elif fault == "extra":
        delta[0, 0] = 1
    elif fault == "unknown":
        delta[0, 0] = np.nan
    s, lab = scenario(delta, GATE, nominal, 1.0, 12)
    if fault == "seam":
        s["regions"][0]["crosses_seam"] = True
    if fault == "none":
        r = metrics(fields, lab, s, True)
        assert len(r) == 2 and all(x["iou"] == 1 and x["signed_peak_error_mm"] == 0 for x in r)
    else:
        with pytest.raises(RuntimeError):
            metrics(fields, lab, s, True)
