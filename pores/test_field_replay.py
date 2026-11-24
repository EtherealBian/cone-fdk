import numpy as np
import pytest
from pore_agreement import compare
from pore_field_replay import load_layers


@pytest.mark.parametrize("model", ["F", "G", "H"])
@pytest.mark.parametrize("bad", ["extra", "shape", "dtype", "nan"])
def test_weight_schema_rejects_corrupt_layers(tmp_path, model, bad):
    d = [22, 16, 16, 16, 16, 1] if model in ("F", "H") else [22, 12, 12, 12, 1]
    weights = {}
    for j, (ci, co) in enumerate(zip(d[:-1], d[1:], strict=True)):
        k = 1 if co == 1 else 3
        weights[f"w{j}"] = np.zeros((co, ci, k, k), "f4")
        weights[f"b{j}"] = np.zeros(co, "f4")
    path = tmp_path / "weights.npz"
    np.savez(path, **weights)
    assert len(load_layers(path, model)) == len(d) - 1
    if bad == "extra":
        weights["metadata"] = np.array(1)
    if bad == "shape":
        weights["w0"] = weights["w0"][:1]
    if bad == "dtype":
        weights["b0"] = weights["b0"].astype("f8")
    if bad == "nan":
        weights["w0"].flat[0] = np.nan
    np.savez(path, **weights)
    with pytest.raises(ValueError):
        load_layers(path, model)


def test_all_eight_vote_patterns_both_reference_values():
    v = np.arange(16)
    e, f, g, t = [(v & k) != 0 for k in (8, 4, 2, 1)]
    hist, m, r = compare(e, f, g, t)
    np.testing.assert_array_equal(hist, np.ones((8, 2)))
    np.testing.assert_array_equal(m, (e * 1 + f + g) >= 2)
    assert np.flatnonzero(r != t).tolist() == [1, 14]
    with pytest.raises(ValueError):
        compare(e, f[:-1], g, t)
