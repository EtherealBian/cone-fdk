import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pore_overlap import analyze  # noqa: E402


def test_graph_distinguishes_bridge_from_absent_and_does_not_censor_small_contacts():
    truth = np.zeros((12, 14, 26), bool)
    truth[2:6, 3:7, 3:7] = 1
    truth[2:6, 3:7, 8:12] = 1
    truth[4:8, 7:11, 18:22] = 1
    pred = truth.copy()
    pred[4:8, 7:11, 18:22] = 0
    pred[3, 4, 7] = 1
    rows = analyze(pred, truth)
    assert len(rows) == 3
    joined = [r for r in rows if r[0]]
    assert len(joined) == 2 and joined[0][0] == joined[1][0]
    assert all(r[2:] == (129, 64, 64, True, True, 2, 1) for r in joined)
    assert rows[0][0] == 0 and rows[0][3:7] == (64, 0, False, True)
    assert analyze(truth * False, truth * False) == []
