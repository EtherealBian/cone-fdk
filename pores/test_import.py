import json

import numpy as np
import pytest
from import_stack import convert
from PIL import Image


def test_indexed_import_is_numeric_and_atomic(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    out = tmp_path / "out"
    args = ((1, 1, 2, 2), (9, 11), (1.0, 2.0, 3.0))
    for n in (10, 9):
        Image.fromarray(np.arange(12, dtype="uint16").reshape(3, 4) + 100 * n).save(
            inp / f"ct{n}.tif"
        )
    convert(inp, out, *args)
    m = json.loads((out / "volume.json").read_text())
    assert [s["index"] for s in m["slices"]] == [9, 10]
    assert m["origin_xyz"] == [1, 1, 9] and m["shape_zyx"] == [2, 2, 2]
    np.testing.assert_array_equal(
        np.fromfile(out / "roi.f32", "<f4"), [905, 906, 909, 910, 1005, 1006, 1009, 1010]
    )
    with pytest.raises(FileExistsError):
        convert(inp, out, *args)
    (inp / "ct10.tif").unlink()
    with pytest.raises(ValueError, match="Missing slice"):
        convert(inp, tmp_path / "bad", *args)
    assert not (tmp_path / "bad").exists()
