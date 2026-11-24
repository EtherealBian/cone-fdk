"""Deliberately optimistic reference coverage/volume bound for choosing intact, already matched pores."""

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path

STUDIES = (
    "section-area",
    "axial-raw",
    "census-train",
    "local-connectivity",
    "two-scale",
    "five-plane",
    "small-seed",
    "mass",
    "forward",
    "monotone",
    "orientation",
    "distance-target",
)
ROOT = Path(__file__).resolve().parents[1] / "pores/evidence"


def p95(v):
    if not v:
        return None
    s = sorted(v)
    q, r = divmod(19 * (len(s) - 1), 20)
    return (s[q] * (20 - r) + s[min(q + 1, len(s) - 1)] * r) / 20


def bound(best):
    n = len(best)
    k = (9 * n + 9) // 10
    vals = sorted(e for e in best if e is not None)
    lo = p95(vals[:k]) if k and len(vals) >= k else None
    return dict(
        reference=n,
        needed=k,
        union=len(vals),
        coverage=len(vals) / n if n else None,
        full_union_p95=float(p95(vals)) if vals else None,
        best_p95=float(lo) if lo is not None else None,
        possible=lo is not None and lo <= 15,
    )


def exact(v, n):
    return Fraction(100 * abs(v - n), n)


def process(i, root=ROOT):
    arms = {}
    canonical = {}
    evidence = {}
    seen = {}
    reference = None
    ref_hash = None
    csv_by_hash = {}
    for name in STUDIES:
        p = root / name / f"sample{i}.json"
        raw = p.read_bytes()
        d = json.loads(raw)
        evidence[name] = hashlib.sha256(raw).hexdigest()
        assert d["specimen"] == i
        sh = d.get("reference_digest", d.get("reference_mask"))
        assert sh and (ref_hash is None or sh == ref_hash)
        ref_hash = sh
        if "origin_xyz" in d:
            assert d["origin_xyz"] == [360, 376, 256] and d["shape_zyx"] == [96, 256, 256]
        for k, a in d["arms"].items():
            key = f"{name}/{k}"
            ref = {int(j): x["reference_voxels"] for j, x in a["references"].items()}
            assert reference is None or reference == ref
            reference = ref
            hit = {
                int(j): x["voxels"] for j, x in a["references"].items() if x["pred"] is not None
            }
            assert (
                len(hit) == a["objects"]["matches"]
                and len(ref) == a["objects"]["eligible_reference"]
                and all(x >= 64 for x in hit.values())
            )
            sig = a["mask_u8_sha256"]
            # Some studies reuse a signed baseline instead of storing its identical CSV again.
            table = root / name / f"sample{i}-{k}-matches.csv"
            if table.is_file():
                with table.open(newline="") as f:
                    rows = list(csv.DictReader(f))
                assert (
                    len(rows) == len(hit)
                    and {int(y["ref_scipy_label"]): int(y["pred_voxels"]) for y in rows} == hit
                )
                assert all(
                    int(y["ref_voxels"]) == ref[int(y["ref_scipy_label"])]
                    and float(y["iou"]) >= 0.5
                    and abs(
                        float(y["signed_volume_error_pct"])
                        - 100
                        * (int(y["pred_voxels"]) - int(y["ref_voxels"]))
                        / int(y["ref_voxels"])
                    )
                    < 1e-10
                    for y in rows
                )
                csv_by_hash.setdefault(sig, []).append(str(table.relative_to(root).as_posix()))
            if sig in seen:
                assert (
                    seen[sig] == hit
                    and arms[canonical[sig]]["eligible_prediction"]
                    == a["objects"]["eligible_prediction"]
                )
            else:
                seen[sig] = hit
                canonical[sig] = key
            tail = p95([exact(v, ref[j]) for j, v in hit.items()])
            n_pred = a["objects"]["eligible_prediction"]
            assert (tail is None and a["objects"]["p95_abs_volume_error_pct"] is None) or abs(
                float(tail) - a["objects"]["p95_abs_volume_error_pct"]
            ) < 1e-9
            arms[key] = dict(
                mask=sig,
                canonical=canonical[sig],
                eligible_prediction=n_pred,
                eligible_reference=len(ref),
                matches=len(hit),
                p95=float(tail) if tail is not None else None,
                two_criteria=bool(hit) and len(hit) * 10 >= 9 * len(ref) and tail <= 15,
            )
    assert csv_by_hash.keys() == seen.keys()  # never infer a CSV check from the JSON itself
    rows = {}
    best = []
    for j, n in sorted(reference.items()):
        obs = {
            canonical[sig]: dict(predicted=h[j], absolute_error_pct=float(exact(h[j], n)))
            for sig, h in seen.items()
            if j in h
        }
        err = min((exact(x["predicted"], n) for x in obs.values()), default=None)
        best.append(err)
        rows[j] = dict(
            reference_voxels=n,
            masks_matched=len(obs),
            best_pct=float(err) if err is not None else None,
            observations=obs,
        )
    return dict(
        specimen=i,
        reference_digest=ref_hash,
        source_records=evidence,
        input_arms=arms,
        independent_csv_by_unique_mask=csv_by_hash,
        unique_masks=len(seen),
        objects=rows,
        never_matched=[j for j, v in rows.items() if not v["observations"]],
        unmixed_passing=[k for k in canonical.values() if arms[k]["two_criteria"]],
        bound=bound(best),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    for i in (2, 3, 5):
        r = process(i)
        (a.out / f"sample{i}.json").write_text(
            json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf8", newline="\n"
        )
        print(i, r["bound"], r["never_matched"], r["unmixed_passing"])
