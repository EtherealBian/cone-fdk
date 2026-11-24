import numpy as np
import pytest
from pore_curvature_fields import DIR, cpu, field, gpu, scalar


@pytest.mark.parametrize("kind", ["flat", "linear", "sphere", "sheet", "rails"])
def test_curvature_chords_invariance_and_registered_order(kind):
    z, y, x = np.indices((15, 17, 19))
    gen = np.random.default_rng(615)
    raw = {
        "flat": 20000 + z * 0,
        "linear": 1200 + 7 * z + 5 * y + 2 * x,
        "sphere": 1200 + z * z + y * y + x * x,
        "sheet": 1200 + z * z,
        "rails": gen.choice([0, 65535], size=z.shape),
    }[kind].astype("u2")
    f = gpu(raw)
    assert len(DIR) == 13 and np.array_equal(f, cpu(raw))
    for loc in ((0, 0, 0), (2, 3, 4), (4, 6, 8)):
        assert np.array_equal(f[loc], scalar(raw, loc))
    assert np.array_equal(gpu(raw[::-1, :, ::-1]), f[::-1, :, ::-1])
    assert np.array_equal(gpu(raw.transpose(2, 1, 0)), f.transpose(2, 1, 0, 3))
    assert np.array_equal(f, np.concatenate((gpu(raw[:12]), gpu(raw[2:]))))
    if kind in ("flat", "linear"):
        assert not np.any(f)
    if kind == "sphere":
        assert np.all(f == np.repeat([2 * 27 * 6 * r * r for r in (1, 2, 4)], 5))
    if kind == "sheet":
        assert not np.any(f[..., ::5]) and np.all(
            f[..., 4::5] == [2 * 27 * 6 * r * r for r in (1, 2, 4)]
        )


@pytest.mark.parametrize(
    "i,a,b", [(4, 160, 164), (6, 256, 260), (2, 608, 612), (3, 183, 185), (5, 400, 404)]
)
def test_curvature_rejects_unregistered_before_source_reads(i, a, b):
    with pytest.raises(ValueError, match="Unregistered"):
        field(None, i, a, b)
