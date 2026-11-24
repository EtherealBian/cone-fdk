"""Three-dimensional directional curvature: calibration-only fit and raw-only prediction."""

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np
from pore_axial_raw import (
    DEV,
    PTX,
    ROOT,
    SHAPE,
    Denoise,
    Z,
    digest,
    features,
    sha,
    stack,
    token,
    verified,
    write,
)
from pore_curvature_fields import field
from pore_local_profile import identity as parent_identity
from pore_local_profile import library_score, load_model, profiles
from pore_profile_gpu import gpu_score, pack
from pore_region import EV, read
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ARMS = ("profile", "curvature")
BASE = EV / "local-profile"
E = EV / "volume-curvature"


def identity():
    names = (
        "pores/VOLUME-CURVATURE-PILOT.md",
        "pores/volume_curvature.cu",
        "scripts/pore_curvature_fields.py",
        "scripts/pore_volume_curvature.py",
        "scripts/pore_volume_curvature_score.py",
    )
    return dict(
        parent=parent_identity(),
        seal=sha(BASE / "predictions.json"),
        ptx=sha(PTX / "volume-curvature.ptx"),
        files={
            n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in names
        },
    )


def fit(cache, feature_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    ident = identity()
    parent = read(BASE / "predictions.json")
    spec = {}
    xx = []
    yy = []
    for i in DEV:
        f = feature_dir / f"cal-{i}.npy"
        r = parent["specimens"][str(i)]
        assert sha(f) == r["cal_features"]
        x = np.load(f, mmap_mode="r", allow_pickle=False)
        assert x.shape == (24, 10, 272, 272)
        seg, ss = verified(cache, i, "segmented", False)
        assert ss["sha256"] == r["segmented"]
        t = stack(seg, i, "segmented", 160, 184)
        assert digest(t) == r["cal_mask"]
        ff = np.concatenate([profiles(x[z], z) for z in range(24)])
        y = np.concatenate([t[z, z % 2 :: 2, (z // 2) % 2 :: 2].ravel() for z in range(24)])
        assert len(ff) == len(y) == 24 * 128**2
        assert digest(ff) == r["profiles"] and digest(y) == r["targets"]
        raw, rs = verified(cache, i, "raw", False)
        assert rs["sha256"] == r["raw"]
        cur = []
        for a in range(160, 184, 4):
            cf = field(raw, i, a, a + 4)
            for j, q in enumerate(cf, a - 160):
                cur.append(q[j % 2 :: 2, (j // 2) % 2 :: 2].reshape(-1, 15))
        c = np.concatenate(cur)
        assert c.shape == (len(ff), 15)
        xx.append(np.concatenate([ff, c], axis=1))
        yy.append(y)
        spec[str(i)] = dict(
            raw=r["raw"],
            segmented=r["segmented"],
            cal_features=r["cal_features"],
            cal_mask=r["cal_mask"],
            rows=len(y),
            void=int(y.sum()),
            profiles=digest(ff),
            curvature_cal=digest(c),
            targets=digest(y),
        )
    x = np.concatenate(xx)
    y = np.concatenate(yy)
    del xx, yy
    fits = {"profile": parent["fits"]["profile"]}
    shutil.copyfile(BASE / "profile.npz", out / "profile.npz")
    assert sha(out / "profile.npz") == fits["profile"]["weights"]
    for k, d in [("curvature", 61)]:
        h = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=96,
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=32,
            l2_regularization=1,
            max_bins=127,
            early_stopping=False,
            random_state=231,
        )
        with threadpool_limits(limits=4):
            h.fit(x[:, :d], y)
        f = out / f"{k}.npz"
        np.savez_compressed(
            f,
            intercept=h._baseline_prediction.ravel(),
            **{f"n{j}": g[0].nodes for j, g in enumerate(h._predictors)},
        )
        b, nodes = load_model(f)
        arr = pack(nodes, d)
        v = x[::31, :d]
        expected = h.decision_function(v)
        cpu = library_score(v, nodes, b)
        native = gpu_score(v, arr, b)
        assert (
            np.array_equal(expected, cpu)
            and np.array_equal(expected, native)
            and np.array_equal(h.predict(v), native > 0)
        )
        fits[k] = dict(
            weights=sha(f), trees=h.n_iter_, features=d, nodes=len(arr[0]), cal_replay=len(v)
        )
        print(k, fits[k], flush=True)
    write(
        out / "fit.json",
        dict(identity=ident, pipeline_id=token(ident), specimens=spec, fits=fits),
    )


def predict(cache, out):
    doc = read(out / "fit.json")
    ident = identity()
    assert doc["identity"] == ident and doc["pipeline_id"] == token(ident)
    models = {}
    old = read(BASE / "predictions.json")
    for k, s in doc["fits"].items():
        f = out / f"{k}.npz"
        assert sha(f) == s["weights"]
        b, nodes = load_model(f)
        models[k] = (b, nodes, pack(nodes, s["features"]), s["features"])
    for i in DEV:
        r = doc["specimens"][str(i)]
        raw, rs = verified(cache, i, "raw", False)
        assert rs["sha256"] == r["raw"]
        masks = {k: np.empty(SHAPE, bool) for k in ARMS}
        errors = dict.fromkeys(ARMS, 0.0)
        flips = dict.fromkeys(ARMS, 0)
        hh = hashlib.sha256()
        hcur = hashlib.sha256()
        with Denoise(PTX / "search-support.ptx") as denoise:
            for off in range(0, 96, 4):
                x = features(raw, i, Z[0] + off, Z[0] + off + 4, denoise)
                hh.update(np.ascontiguousarray(x).view("u1"))
                c = field(raw, i, Z[0] + off, Z[0] + off + 4)
                hcur.update(c.view("u1"))
                for z, f in enumerate(x, off):
                    v = np.concatenate([profiles(f), c[z - off].reshape(-1, 15)], axis=1)
                    for k, (b, nodes, a, d) in models.items():
                        native = gpu_score(v[:, :d], a, b)
                        cpu = library_score(v[:, :d], nodes, b)
                        assert np.isfinite(native).all() and np.isfinite(cpu).all()
                        m = native > 0
                        masks[k][z] = m.reshape(SHAPE[1:])
                        flips[k] += int(np.count_nonzero(m != (cpu > 0)))
                        errors[k] = max(errors[k], float(abs(native - cpu).max()))
                if off % 24 == 0:
                    print(i, off, flush=True)
        r["curvature_f32"] = hcur.hexdigest()
        r["integer_features_checked"] = int(np.prod(SHAPE)) * 15
        r["features_f32"] = hh.hexdigest()
        assert r["features_f32"] == old["specimens"][str(i)]["features_f32"]
        r["masks"] = {}
        for k, m in masks.items():
            f = out / f"sample{i}-{k}.npz"
            np.savez_compressed(f, mask=m)
            r["masks"][k] = dict(
                file=sha(f),
                mask=digest(m),
                cpu_gpu_differences=flips[k],
                max_score_difference=errors[k],
            )
        assert (
            not any(flips.values())
            and r["masks"]["profile"] == old["specimens"][str(i)]["masks"]["profile"]
        )
        print(i, r["masks"], flush=True)
    doc.update(shape_zyx=list(SHAPE), origin_xyz=[360, 376, Z[0]])
    write(out / "predictions.json", doc)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for k in ("fit", "predict"):
        a = sub.add_parser(k)
        a.add_argument("--cache", type=Path, required=True)
        a.add_argument("--out", type=Path, required=True)
        if k == "fit":
            a.add_argument("--feature-dir", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "fit":
        fit(a.cache, a.feature_dir, a.out)
    else:
        predict(a.cache, a.out)
