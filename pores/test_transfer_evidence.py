"""The published transfer decision must follow the registered folds and the per-ID ledger."""

import json

import numpy as np
from pore_transfer import DEV, ROOT, fit, screen
from pore_transfer_report import EVIDENCE, REPORT, render


def test_published_transfer_records_agree_with_unseen_slab_seal_and_counts():
    manifest = json.loads((EVIDENCE / "baseline.json").read_text())
    assert manifest["shape_zyx"] == [96, 256, 256] and manifest["z"] == [256, 352]
    rec = {i: json.loads((EVIDENCE / f"sample{i}.json").read_text()) for i in DEV}
    for i, r in rec.items():
        old = json.loads((ROOT / f"pores/evidence/candidate-e/sample{i}.json").read_text())
        assert manifest["model"] == old["pipeline_id"]
        assert (
            r["sealed"] == manifest["specimens"][str(i)] and len(r["sealed"]["probability"]) == 64
        )
        assert r["source_xyz"] == [360, 376, 256] and r["shape_zyx"] == manifest["shape_zyx"]
        donors = {str(k): v["donor_pairs_for_other_folds"] for k, v in rec.items() if k != i}
        assert r["trained_on"] == {k: len(v) for k, v in donors.items()}
        np.testing.assert_allclose(fit(donors), r["coefficients"], atol=1e-15)
        for stage in ("base", "arm"):
            a = r[stage]
            rows = a["references"]
            matched = [v for v in rows.values() if v["pred"] is not None]
            o = a["objects"]
            assert len(rows) == o["eligible_reference"] and len(matched) == o["matches"]
            pct = [
                100 * abs(v["voxels"] - v["reference_voxels"]) / v["reference_voxels"]
                for v in matched
            ]
            if pct:
                np.testing.assert_allclose(
                    np.percentile(pct, 95), o["p95_abs_volume_error_pct"], rtol=1e-12
                )
            else:
                assert o["p95_abs_volume_error_pct"] is None
        sc = screen(r["base"], r["arm"], r["trained_on"])
        sc["shared_ids"] = [int(k) for k in sc["shared_ids"]]
        assert sc == r["screen"]
    assert REPORT.read_text(encoding="utf8") == render(EVIDENCE)
