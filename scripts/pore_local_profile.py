"""Raw local intensity profiles, separate calibration-only fit and raw-only transfer inference."""

import argparse
import hashlib
from importlib.metadata import version
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
from pore_profile_gpu import gpu_score, pack
from pore_region import EV, read
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from threadpoolctl import threadpool_limits

ARMS = ("center", "profile")
E = EV / "local-profile"
CHANNEL = (0, 5, 6, 8)
RAD = (1, 2, 4)
ORDER = (1, 4, 7)


def identity():
    files = (
        "pores/LOCAL-PROFILE-PILOT.md",
        "pores/local_profile.cu",
        "scripts/pore_local_profile.py",
        "scripts/pore_profile_gpu.py",
        "scripts/pore_profile_score.py",
        "scripts/pore_axial_raw.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_section_score.py",
        "scripts/pore_transfer.py",
        "pores/validate.py",
        "pores/pore_check.py",
        "pores/local_model.py",
        "pores/search_support.cu",
        "scripts/pore_search_support.py",
    )
    return dict(
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in files
        },
        compiled={
            k: sha(PTX / k)
            for k in ("local-profile.ptx", "search-support.ptx", "pores.ptx", "cone-pores.exe")
        },
        versions={k: version(k) for k in ("numpy", "scipy", "scikit-learn")},
        parents={
            k: sha(EV / k / "predictions.json")
            for k in ("axial-raw", "orientation", "census-train")
        },
    )


def profiles(x, off=None):
    if x.ndim != 3 or x.dtype != np.float32 or x.shape[0] != 10 or min(x.shape[1:]) <= 16:
        raise ValueError("Expected ten float fields and an eight-pixel real halo")
    h, w = (s - 16 for s in x.shape[1:])
    py, px, step = (off % 2, (off // 2) % 2, 2) if off is not None else (0, 0, 1)

    def crop(c, dy, dx):
        return x[c, 8 + dy + py : 8 + dy + h : step, 8 + dx + px : 8 + dx + w : step]

    vectors = [crop(c, 0, 0) for c in range(10)]
    for c in CHANNEL:
        for r in RAD:
            v = np.stack([crop(c, y, u) for y in (-r, 0, r) for u in (-r, 0, r)], axis=-1)
            v.sort(axis=-1)
            vectors.extend(v[..., q] - vectors[c] for q in ORDER)
    field = np.stack(vectors, axis=-1)
    # Scalar independent sorting and direct point reads at corner/center/opposite corner.
    for y, u in (
        (0, 0),
        (field.shape[0] // 2, field.shape[1] // 2),
        (field.shape[0] - 1, field.shape[1] - 1),
    ):
        iy = 8 + py + step * y
        ix = 8 + px + step * u
        v = list(x[:, iy, ix])
        for c in CHANNEL:
            for r in RAD:
                vals = sorted(x[c, iy + a, ix + b] for a in (-r, 0, r) for b in (-r, 0, r))
                v.extend(vals[q] - x[c, iy, ix] for q in ORDER)
        assert np.array_equal(field[y, u], v)
    assert np.isfinite(field).all()
    return field.reshape(-1, field.shape[-1])


def library_score(x, nodes, base):
    # The original sklearn numeric TreePredictor; compare this export against the fitted estimator too.
    f = np.asarray(x, dtype="f8", order="C")
    v = np.full(len(x), base, dtype="f8")
    empty = np.empty((0, 8), "u4")
    fmap = np.zeros(x.shape[1], "u4")
    for a in nodes:
        v += TreePredictor(a, empty, empty).predict(f, empty, fmap, 4)
    return v


def load_model(f):
    with np.load(f, allow_pickle=False) as doc:
        base = doc["intercept"].item()
        nodes = [doc[f"n{i}"] for i in range(len(doc.files) - 1)]
    return base, nodes


def fit(cache, feature_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    ident = identity()
    parent = read(EV / "axial-raw" / "predictions.json")
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
        xx.append(ff)
        yy.append(y)
        spec[str(i)] = dict(
            raw=r["raw"],
            segmented=r["segmented"],
            cal_features=r["cal_features"],
            cal_mask=r["cal_mask"],
            rows=len(y),
            void=int(y.sum()),
            profiles=digest(ff),
            targets=digest(y),
        )
    x = np.concatenate(xx)
    y = np.concatenate(yy)
    del xx, yy
    fits = {}
    for k, d in zip(ARMS, (10, 46), strict=True):
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
    old = read(EV / "orientation" / "predictions.json")
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
        with Denoise(PTX / "search-support.ptx") as denoise:
            for off in range(0, 96, 4):
                x = features(raw, i, Z[0] + off, Z[0] + off + 4, denoise)
                hh.update(np.ascontiguousarray(x).view("u1"))
                for z, f in enumerate(x, off):
                    v = profiles(f)
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
