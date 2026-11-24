"""Score the three publicly sealed F development fields; run independent source replay."""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pores"))
from audit import replay
from freeze import token
from pore_check import native, reference
from pore_spatial_predict import DEV, NAME, SEAL, SHAPE, Z, digest, sha, stack, verified
from review import write_review
from validate import ORIGIN, SPEC, final_gates, objects, score


def run(baseline, cache, out, *, seal=SEAL, method=NAME, field_key="logits"):
    if field_key not in ("logits", "field"):
        raise ValueError("Unknown sealed field")
    if not __debug__:
        raise ValueError("Do not run with -O")
    locked = json.loads(seal.read_text())
    local = json.loads((baseline / "predictions.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    if (
        any(out.iterdir())
        or locked != local
        or locked["pipeline_id"] != token(locked["identity"])
        or locked["origin_xyz"] != list(ORIGIN)
        or locked["shape_zyx"] != list(SHAPE)
        or set(locked["specimens"]) != {str(i) for i in DEV}
    ):
        raise ValueError("Unexpected output or manifest")
    for i in DEV:
        r = locked["specimens"][str(i)]
        s = baseline / f"sample{i}.f32"
        w = baseline / f"sample{i}-weights.npz"
        if (
            sha(s) != r[field_key]
            or sha(w) != r["weights"]
            or s.stat().st_size != 4 * np.prod(SHAPE)
        ):
            raise ValueError("Unsealed logits or weights")
        ll = np.memmap(s, mode="r", dtype="<f4", shape=SHAPE)
        if not np.isfinite(ll).all() or digest(ll >= 0) != r["mask"]:
            raise ValueError("Unsealed or nonfinite mask")
    reports = {}
    for i in DEV:
        stem = f"sample{i}"
        r = locked["specimens"][str(i)]
        predlog = np.memmap(baseline / (stem + ".f32"), mode="r", dtype="<f4", shape=SHAPE)
        mask = predlog >= 0
        spacing = (SPEC[i][2],) * 3
        raw, rs = verified(cache, i, "raw", False)
        seg, ss = verified(cache, i, "segmented", False)
        if (rs["sha256"], ss["sha256"]) != (r["raw"], r["segmented"]):
            raise ValueError("Different raw/reference sources")
        table = out / (stem + "-pores.csv")
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            src = d / "signed.f32"
            (-predlog).astype("<f4").tofile(src)
            st = native(src, table, SHAPE, spacing, 0, 6, 1000, "none", 4)
            lab, rows = reference(mask, 6, spacing, table, st)
            second = native(src, d / "second.csv", SHAPE, spacing, 0, 6, 37, "none", 64)
            if (st["slab_depth"], second["slab_depth"]) != (8, 37) or (
                d / "second.csv"
            ).read_bytes() != table.read_bytes():
                raise ValueError("Slab partition mismatch")
        truth = stack(seg, i, "segmented", *Z)
        m = score(mask, truth)
        o = objects(lab, truth, rows, out / (stem + "-matches.csv"))
        g = final_gates(m, o)
        view = out / (stem + "-review.html")
        meta = write_review(
            view, i, stack(raw, i, "raw", *Z), truth, lab, rows, ORIGIN, spacing[0]
        )
        report = dict(
            specimen=i,
            role="development",
            method=method,
            pipeline_id=locked["pipeline_id"],
            confirmation_manifest=None,
            source=dict(raw=rs, segmented=ss),
            origin_xyz=ORIGIN,
            calibration_z=[160, 184],
            evaluation_z=list(Z),
            **(
                dict(logit_threshold=0, probability_threshold=0.5)
                if field_key == "logits"
                else dict(
                    decision_margin_threshold=0, threshold_rule="registered digital Bernsen"
                )
            ),
            metrics=m,
            objects=o,
            final_gates=g,
            final_passed=all(g.values()),
            table_sha256=sha(table),
            review_sha256=sha(view),
            review=meta,
            same_mask_scipy_rows=len(rows),
            second_slab_depth=37,
            native=st,
            seal=r,
            reference_mask=digest(truth),
        )
        (out / (stem + ".json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        name, checked = replay(out, i, cache)
        if name != method or checked != g:
            raise ValueError("Independent replay disagrees")
        reports[str(i)] = dict(
            gates=sum(g.values()),
            p95=o["p95_abs_volume_error_pct"],
            median=o["median_abs_volume_error_pct"],
            matched=o["matches"],
            prediction=o["eligible_prediction"],
            reference=o["eligible_reference"],
            failed=[k for k, v in g.items() if not v],
            cpu_gpu_flips=r["cpu_gpu_decision_disagreements"],
        )
        print(method, "replay", i, reports[str(i)], flush=True)
    (out / "summary.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["baseline", "cache", "out"]:
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    run(a.baseline, a.cache, a.out)
