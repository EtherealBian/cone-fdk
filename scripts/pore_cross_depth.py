"""Replicate two frozen raw segmentation models at an isolated development depth."""

import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pore_axial_raw import (
    DEV,
    PTX,
    ROOT,
    SETTINGS,
    SHAPE,
    Denoise,
    digest,
    features,
    sha,
    token,
    verified,
    write,
)
from pore_axial_score import read_mask
from pore_local_profile import (
    ARMS,
    library_score,
    load_model,
    profiles,
)
from pore_local_profile import (
    E as MODEL,
)
from pore_local_profile import (
    identity as model_identity,
)
from pore_profile_gpu import gpu_score, pack
from pore_region import EV, read
from pore_transfer import SPEC
from scipy import ndimage as ndi

E = EV / "cross-depth"
Z = (608, 704)
CHECK = (256, 260)
XYZ = (360, 376, Z[0])


def identity():
    keys = (
        "scripts/pore_cross_depth.py",
        "scripts/pore_cross_depth_score.py",
        "pores/CROSS-DEPTH-REPLICATION.md",
        "scripts/pore_axial_score.py",
        "scripts/pore_region.py",
        "pores/denoise.py",
        "scripts/pore_monotone_score.py",
        "scripts/pore_section_score.py",
        "scripts/pore_transfer.py",
        "pores/validate.py",
        "pores/pore_check.py",
    )
    return dict(
        parent=sha(MODEL / "predictions.json"),
        parent_code=model_identity(),
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in keys
        },
    )


def eligible(i, a, b):
    if i not in DEV or not any(x <= a < b <= y for x, y in (CHECK, Z)):
        raise ValueError("Outside registered development intervals")


def median27(v):
    if v.dtype != np.uint16 or v.ndim != 3 or v.shape[0] < 3:
        raise ValueError("Expected raw native uint16 planes with two real axial neighbors")
    pad = np.pad(v, ((0, 0), (1, 1), (1, 1)), mode="edge")
    h, w = v.shape[1:]
    n = len(v) - 2
    win = np.stack(
        [
            pad[z : z + n, y : y + h, x : x + w]
            for z in range(3)
            for y in range(3)
            for x in range(3)
        ],
        axis=-1,
    )
    win.partition(13, axis=-1)
    return np.ascontiguousarray(win[..., 13])


def field(path, i, a, b, gpu):
    eligible(i, a, b)
    prefix = f"set1sample{i}raw"
    with ZipFile(path) as archive:
        planes = []
        for z in range(a - 3, b + 3):
            with Image.open(io.BytesIO(archive.read(f"{prefix}/{prefix}_{z:04d}.tif"))) as im:
                assert im.size == SPEC[i][:2]
                planes.append(np.asarray(im)[350:658, 334:642].copy())
    raw = np.stack(planes)
    n = b - a
    assert raw.dtype == np.uint16 and raw.shape == (n + 6, 308, 308)
    med = median27(raw[2:-2])
    scale, shift, ct = SETTINGS[i]
    mapped = (gpu(med, 30 / scale) + shift) * scale
    u = np.clip(np.floor(mapped + 0.5), 0, 255).astype("i2")
    y, x = np.ogrid[-5:6, -5:6]
    footprint = (x * x + y * y <= 26)[None]
    lo = ndi.minimum_filter(u, footprint=footprint)
    hi = ndi.maximum_filter(u, footprint=footprint)
    mid = (hi + lo) // 2
    span = hi - lo
    prior = np.clip(np.where(span < ct, 127.5 - mid, mid - u - 0.5), -4, 4)
    r = (raw[:, 18:-18, 18:-18].astype("f4") + shift) * scale
    c = r[3 : 3 + n]
    ch = [(c - 128) / 48]
    for step in (1, 3):
        lh, rh = r[3 - step : 3 - step + n], r[3 + step : 3 + step + n]
        ch.extend(((0.5 * (lh + rh) - c) / 24, np.abs(lh - rh) / 48))
    sl = np.s_[:, 18:-18, 18:-18]
    ch.extend(
        (
            ((med[sl].astype("f4") + shift) * scale - 128) / 48,
            (mapped[sl] - 128) / 48,
            (span[sl] - ct) / 6,
            (mid[sl] - u[sl]) / 4,
            prior[sl],
        )
    )
    f = np.stack(ch, axis=1).astype("f4")
    assert f.shape == (n, 10, 272, 272) and np.isfinite(f).all()
    return f


def classify(f, models):
    v = profiles(f)
    masks = {}
    errors = {}
    for k, (inter, nodes, arr, d) in models.items():
        native = gpu_score(v[:, :d], arr, inter)
        cpu = library_score(v[:, :d], nodes, inter)
        if not (
            np.isfinite(native).all()
            and np.isfinite(cpu).all()
            and np.array_equal(native > 0, cpu > 0)
        ):
            raise ValueError("CPU/CUDA classifications differ or nonfinite")
        masks[k] = (native > 0).reshape(SHAPE[1:])
        errors[k] = float(np.abs(native - cpu).max())
    return masks, errors


def predict(cache, out):
    out.mkdir(parents=True, exist_ok=False)
    src = read(MODEL / "predictions.json")
    ident = identity()
    assert src["identity"] == ident["parent_code"] and set(src["specimens"]) == set("235")
    models = {}
    weights = {}
    for k in ARMS:
        p = MODEL / f"{k}.npz"
        v = src["fits"][k]
        weights[k] = sha(p)
        assert weights[k] == v["weights"] and v["trees"] == 96
        base, nodes = load_model(p)
        models[k] = (base, nodes, pack(nodes, v["features"]), v["features"])
    specimens = {}
    for i in DEV:
        r = src["specimens"][str(i)]
        raw, rs = verified(cache, i, "raw", False)
        assert rs["sha256"] == r["raw"]
        orig = {k: read_mask(MODEL, i, k, r["masks"][k])[:4] for k in ARMS}
        target = {k: np.empty(SHAPE, bool) for k in ARMS}
        hh = hashlib.sha256()
        errors = dict.fromkeys(ARMS, 0.0)
        with Denoise(PTX / "search-support.ptx") as gpu:
            check = field(raw, i, *CHECK, gpu)
            assert np.array_equal(check, features(raw, i, *CHECK, gpu))
            for j, c in enumerate(check):
                ms, _ = classify(c, models)
                assert all(np.array_equal(ms[k], orig[k][j]) for k in ARMS)
            for a in range(*Z, 4):
                f = field(raw, i, a, a + 4, gpu)
                hh.update(np.ascontiguousarray(f).view("u1"))
                for j, c in enumerate(f, a - Z[0]):
                    ms, err = classify(c, models)
                    for k in ARMS:
                        target[k][j] = ms[k]
                        errors[k] = max(errors[k], err[k])
                if (a - Z[0]) % 24 == 0:
                    print(i, a, flush=True)
        records = {}
        for k, m in target.items():
            p = out / f"sample{i}-{k}.npz"
            np.savez_compressed(p, mask=m)
            records[k] = dict(
                file=sha(p), mask=digest(m), cpu_gpu_differences=0, max_score_difference=errors[k]
            )
        specimens[str(i)] = dict(
            raw=r["raw"],
            segmented=r["segmented"],
            spacing_xyz_um=[SPEC[i][2]] * 3,
            fields=hh.hexdigest(),
            check_fields=digest(check),
            check_fields_count=check.size,
            check_mask_comparisons=sum(x.size for x in orig.values()),
            masks=records,
        )
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            weights=weights,
            origin_xyz=list(XYZ),
            shape_zyx=list(SHAPE),
            specimens=specimens,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cache", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    predict(a.cache, a.out)
