"""Train each calibration detector without that specimen; transfer only the pooled XY gate."""

import argparse
import hashlib
import platform
from pathlib import Path
from shutil import copyfile

import numpy as np
import sklearn
from pore_axial_raw import (
    DEV,
    PTX,
    ROOT,
    SETTINGS,
    SHAPE,
    Denoise,
    Z,
    digest,
    features,
    sha,
    token,
    write,
)
from pore_chord_fraction import sources
from pore_curvature_fields import field
from pore_face_profile import BASE, CAL
from pore_local_profile import library_score, load_model, profiles
from pore_profile_gpu import gpu_score, pack
from pore_region import EV, read
from pore_section_gate import E as PREV
from pore_section_gate import choose, correction, load_source, sections, solve
from pore_section_gate import identity as old_identity
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

E = EV / "crossfit-gate"
CAL_SHAPE = (CAL[1] - CAL[0], *SHAPE[1:])


def identity():
    names = (
        "pores/CROSSFIT-GATE-PILOT.md",
        "scripts/pore_crossfit_gate.py",
        "scripts/pore_crossfit_gate_score.py",
        "scripts/pore_volume_curvature.py",
        "scripts/pore_compare_score_runs.py",
    )
    return dict(
        parent=old_identity(),
        previous=sha(PREV / "predictions.json"),
        files={
            n: hashlib.sha256((ROOT / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for n in names
        },
        python=platform.python_version(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
    )


def donors(i):
    if i not in DEV:
        raise ValueError("Only development specimens can be held out")
    return tuple(j for j in DEV if j != i)


def training_arrays(i, bank):
    pairs = [bank[j] for j in donors(i)]
    return tuple(np.concatenate(v) for v in zip(*pairs, strict=True))


def quarter(a):
    return np.concatenate(
        [v[j % 2 :: 2, (j // 2) % 2 :: 2].reshape(-1, *a.shape[3:]) for j, v in enumerate(a)]
    )


def inputs(cache):
    full, sampled, data = {}, {}, {}
    for i in DEV:
        path, raw = load_source(cache, i, CAL, "raw")
        _, truth = load_source(cache, i, CAL, "segmented")
        v = np.empty((*CAL_SHAPE, 61), "f4")
        with Denoise(PTX / "search-support.ptx") as den:
            for z in range(*CAL, 4):
                a = features(path, i, z, z + 4, den)
                c = field(path, i, z, z + 4)
                for j in range(4):
                    q = v[z + j - CAL[0]]
                    q[..., :46] = profiles(a[j]).reshape(256, 256, 46)
                    q[..., 46:] = c[j]
        x, y = quarter(v), quarter(truth)
        s = read(BASE / "predictions.json")["specimens"][str(i)]
        assert (digest(x[:, :46]), digest(x[:, 46:]), digest(y), digest(truth)) == (
            s["profiles"],
            s["curvature_cal"],
            s["targets"],
            s["cal_mask"],
        )
        full[i], sampled[i], data[i] = v, (x, y), (raw, truth)
        print("source feature/target digests verified:", i, flush=True)
    return full, sampled, data


def train(i, bank, out):
    x, y = training_arrays(i, bank)
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
        h.fit(x, y)
    f = out / f"fold{i}.npz"
    np.savez_compressed(
        f,
        intercept=h._baseline_prediction.ravel(),
        **{f"n{j}": g[0].nodes for j, g in enumerate(h._predictors)},
    )
    bias, nodes = load_model(f)
    arr = pack(nodes, 61)
    sub = x[::31]
    native = gpu_score(sub, arr, bias)
    assert np.array_equal(native, h.decision_function(sub)) and np.array_equal(
        native, library_score(sub, nodes, bias)
    )
    return (bias, nodes, arr), dict(
        held_out=i,
        donors=list(donors(i)),
        samples=len(y),
        training_x=digest(x),
        training_y=digest(y),
        weights=sha(f),
    )


def infer(v, model):
    bias, nodes, arr = model
    out = []
    for plane in v:
        x = plane.reshape(-1, 61)
        native = gpu_score(x, arr, bias)
        assert np.isfinite(native).all() and np.array_equal(native, library_score(x, nodes, bias))
        out.append((native > 0).reshape(plane.shape[:2]))
    return np.stack(out)


def run(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    ident = identity()
    full, bank, data = inputs(cache)
    pop, cal = {}, {}
    for i in DEV:
        model, fold = train(i, bank, out)
        raw, truth = data[i]
        m = infer(full.pop(i), model)
        f = out / f"sample{i}-oof.npz"
        np.savez_compressed(f, mask=m)
        rows = sections(raw, m, SETTINGS[i][0], truth=truth)
        pop[str(i)] = rows
        good = [r for r in rows if "x" in r]
        cal[str(i)] = dict(
            **fold,
            ref_fg=int(truth.sum()),
            pred_fg=int(m.sum()),
            fp=int((m & ~truth).sum()),
            fn=int((~m & truth).sum()),
            raw=digest(raw),
            reference=digest(truth),
            mask=digest(m),
            file=sha(f),
            sections=len(good),
            parents=len({r["parent"] for r in good}),
            clipped_parents=len({r["parent"] for r in good if r["clipped"]}),
        )
        print("fold", i, cal[str(i)], flush=True)
    del bank, data
    w = solve(pop)
    for rows in pop.values():
        for r in rows:
            r["correction_1024"] = correction(r, w)
    f = out / "fit.json"
    write(f, dict(weights=w.tolist(), calibration=cal, rows=pop))
    spec = {}
    for i in DEV:
        (m, end), g, src = sources(i)
        _, raw = load_source(cache, i, Z, "raw")
        rows = sections(raw, m, SETTINGS[i][0], eligible=set(map(int, g["objects"])))
        pred, choices = choose(m, end, g["objects"], rows, w)
        ledger = out / f"sample{i}-choices.json"
        write(ledger, dict(objects=choices, sections=rows))
        p = out / f"sample{i}-crossfit.npz"
        np.savez_compressed(p, mask=pred)
        spec[str(i)] = dict(
            raw=src[0],
            segmented=src[1],
            parents=[digest(m), digest(end)],
            ledger=sha(ledger),
            masks=dict(crossfit=dict(file=sha(p), mask=digest(pred))),
        )
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            shape_zyx=list(SHAPE),
            origin_xyz=[360, 376, Z[0]],
            fit=sha(f),
            specimens=spec,
        ),
    )


def names():
    return {
        "fit.json",
        "predictions.json",
        *(f"fold{i}.npz" for i in DEV),
        *(f"sample{i}-{k}" for i in DEV for k in ("choices.json", "crossfit.npz", "oof.npz")),
    }


def seal(first, second, out):
    if (
        first.resolve() == second.resolve()
        or out.exists()
        or any({p.name for p in folder.iterdir()} != names() for folder in (first, second))
    ):
        raise ValueError("Need two separate, complete runs and a new output")
    hashes = {}
    for n in sorted(names()):
        if (first / n).read_bytes() != (second / n).read_bytes():
            raise ValueError(f"Replay differs: {n}")
        hashes[n] = sha(first / n)
    out.mkdir(parents=True, exist_ok=False)
    for n in hashes:
        copyfile(first / n, out / n)
    write(out / "prediction-replay.json", hashes)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, keys in [("predict", ("cache", "out")), ("seal", ("first", "second", "out"))]:
        p = sub.add_parser(cmd)
        for k in keys:
            p.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "predict":
        run(a.cache, a.out)
    else:
        seal(a.first, a.second, a.out)
