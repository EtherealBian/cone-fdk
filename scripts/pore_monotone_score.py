"""Score the two locked boundary interventions on the previously seen transfer slab."""

import argparse
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, SHAPE, Z, digest, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_forward import E
from pore_monotone import ARMS, OUT, identity, load, support
from pore_section_score import compare
from pore_transfer import label, measure
from scipy import ndimage as ndi


def comparison(old, new):
    c = compare(old, new)
    a, b = [
        {int(k) for k, v in arm["references"].items() if v["pred"] is not None}
        for arm in (old, new)
    ]
    return dict(**c, gained=sorted(b - a), lost=sorted(a - b))


def run(cache, out):
    seal = load(OUT, "predictions.json")
    prior = load(EVID, "predictions.json")
    ident = identity()
    assert (
        seal["identity"] == ident
        and seal["pipeline_id"] == token(ident)
        and set(seal["specimens"]) == set("235")
        and seal["shape_zyx"] == list(SHAPE)
        and seal["origin_xyz"] == [360, 376, 256]
    )
    for i in DEV:
        r = seal["specimens"][str(i)]
        for k in ARMS:
            read_mask(OUT, i, k, r["masks"][k])
            assert r["masks"][k]["cpu_gpu_differences"] == r["masks"][k]["off_support"] == 0
        assert all(t["gain_numerator"] >= 0 for t in r["colors"]) and r["per_voxel_gain"] >= 0
    out.mkdir(parents=True, exist_ok=False)
    summ = {}
    for i in DEV:
        key = str(i)
        r = seal["specimens"][key]
        sp, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        truth = stack(sp, i, "segmented", *Z)
        tx = label(truth)
        prev = load(E, f"sample{i}.json")
        assert digest(truth) == prev["reference_digest"]
        base = read_mask(EVID, i, "census", prior["specimens"][key]["masks"]["census"])
        assert digest(base) == r["census"]
        arms = {k: prev["arms"][k] for k in ("census", "blur")}
        edit = {}
        deep = (base & (ndi.distance_transform_edt(base) > 4)) | (
            ~base & (ndi.distance_transform_edt(~base) > 4)
        )
        for k in ARMS:
            v = read_mask(OUT, i, k, r["masks"][k])
            arms[k] = measure(v, truth, tx, i, out, k)
            delta = v != base
            fp = delta & v & ~truth
            fn = delta & ~v & truth
            edit[k] = dict(
                new_false_positive=int(fp.sum()),
                new_false_negative=int(fn.sum()),
                corrected_false_positive=int(np.sum(delta & ~v & ~truth)),
                corrected_false_negative=int(np.sum(delta & v & truth)),
                deep_false_positive=int(np.sum(deep & fp)),
                deep_false_negative=int(np.sum(deep & fn)),
                off_support=int(np.sum(delta & ~support(base))),
            )
            assert not any(
                edit[k][j] for j in ("deep_false_positive", "deep_false_negative", "off_support")
            )
        comp = {
            k: {
                a: comparison(arms[a], arms[k])
                for a in (
                    ("census", "blur") if k == "band-sync" else ("census", "blur", "band-sync")
                )
            }
            for k in ARMS
        }
        summ[key] = dict(
            primary=comp["band-color"]["census"]["advance"],
            ordering=comp["band-color"]["band-sync"]["advance"],
        )
        write(
            out / f"sample{i}.json",
            dict(
                specimen=i,
                pipeline_id=seal["pipeline_id"],
                reference_digest=digest(truth),
                arms=arms,
                edits=edit,
                comparisons=comp,
            ),
        )
        print(i, summ[key], flush=True)
    write(
        out / "summary.json",
        dict(
            pipeline_id=seal["pipeline_id"],
            advance=all(v["primary"] for v in summ.values()),
            order_helped_on_all=all(v["ordering"] for v in summ.values()),
            specimens=summ,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    run(a.cache, a.out)
