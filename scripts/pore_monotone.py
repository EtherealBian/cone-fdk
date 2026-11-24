"""One fixed-support sweep. Inference only reads raw images and published parent predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pore_axial_raw import DEV, PTX, ROOT, SHAPE, Z, digest, sha, stack, token, verified, write
from pore_axial_score import read_mask
from pore_census_train import EVID
from pore_forward import D, E, kernel
from pore_forward_energy import energy
from pore_forward_inverse import cpu_step, id_inverse
from pore_monotone_gpu import color_gpu
from pore_transfer import FACE
from scipy import ndimage as ndi
from scipy.signal import convolve

OUT = ROOT / "pores/evidence/monotone"
ARMS = ("band-sync", "band-color")


def load(folder, name):
    return json.loads((folder / name).read_bytes())


def identity():
    return dict(
        parent=id_inverse(),
        predictions=sha(E / "predictions.json"),
        ptx=sha(PTX / "pore-monotone.ptx"),
        files={
            k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for k in (
                "pores/MONOTONE-PILOT.md",
                "scripts/pore_monotone.py",
                "scripts/pore_monotone.cu",
                "scripts/pore_monotone_gpu.py",
                "scripts/pore_monotone_score.py",
                "scripts/pore_forward_energy.py",
                "scripts/pore_section_score.py",
            )
        },
    )


def support(m):
    sides = ndi.maximum_filter(m, footprint=FACE) != ndi.minimum_filter(m, footprint=FACE)
    band = np.zeros_like(m)
    sl = np.s_[4:-4, 4:-4, 4:-4]
    band[sl] = sides[sl]
    return band


def color_cpu(base, raw, q, phases):
    # Expand K^T K into a single 9^3 integer stencil rather than reproducing CUDA's nested residual loops.
    _, g = kernel(q)
    corr = convolve(g, g, mode="full", method="direct")
    assert corr.dtype == np.int64 and corr.sum() == D * D
    nz, y, x = base.shape
    p, m = phases
    c = m - p
    drive = ndi.correlate(m - 2 * raw.astype("i4"), g, mode="nearest", output="i8").ravel() * D
    cur = base.ravel().copy()
    gain = np.zeros(cur.shape, "i8")
    coords = np.argwhere(support(base))
    classes = (coords[:, 0] % 5 * 5 + coords[:, 1] % 5) * 5 + coords[:, 2] % 5
    ids = np.ravel_multi_index(coords.T, base.shape)
    offsets = [
        ((a - 4) * y * x + (b - 4) * x + d - 4, int(corr[a, b, d]))
        for a, b, d in np.argwhere(corr != 0)
    ]
    norm = int((g * g).sum())
    trace = []
    for k in range(125):
        i = ids[classes == k]
        back = np.zeros(len(i), "i8")
        for off, w in offsets:
            back += cur[i + off] * w
        n = sum(cur[i + off].astype("i4") for off in (-1, 1, -x, x, -x * y, x * y))
        msign = 2 * cur[i].astype("i8") - 1
        vote = 12 * (drive[i] - c * back) + c * norm * (9 * msign + n - 3)
        trade = -msign * vote
        positive = trade > 0
        gains = trade[positive]
        gain[i[positive]] = gains
        cur[i[positive]] = ~cur[i[positive]]
        trace.append(
            dict(
                color=k, available=len(i), changed=len(gains), gain_numerator=sum(map(int, gains))
            )
        )
    return cur.reshape(base.shape), gain.reshape(base.shape), trace


def predict(cache, out):
    f = load(E, "predictions.json")
    fit = load(E, "fit.json")
    parent = load(EVID, "predictions.json")
    ident = identity()
    assert f["identity"] == id_inverse() and f["pipeline_id"] == token(f["identity"])
    q = fit["quantized"]
    out.mkdir(parents=True, exist_ok=False)
    spec = {}
    for i in DEV:
        r = f["specimens"][str(i)]
        rp, rs = verified(cache, i, "raw", False)
        raw = stack(rp, i, "raw", *Z)
        base = read_mask(EVID, i, "census", parent["specimens"][str(i)]["masks"]["census"])
        assert (
            digest(base) == r["census"]
            and rs["sha256"] == r["raw"]
            and digest(raw) == r["source_crop"]
        )
        band = support(base)
        phases = [r["predicted_phases"][s]["median_x2"] for s in ("pore", "material")]
        p, m = phases
        assert r["predicted_phases"]["valid"]
        sync = read_mask(E, i, "blur", r["masks"]["blur"])
        check, _ = cpu_step(base, raw, q, phases)
        sync = np.where(band, sync, base)
        assert np.array_equal(sync, np.where(band, check, base))
        colored, gg = color_gpu(base, raw, q, phases)
        cpu, cg, trace = color_cpu(base, raw, q, phases)
        assert (
            np.array_equal(cpu, colored)
            and np.array_equal(cg, gg)
            and np.array_equal(gg > 0, colored != base)
        )
        u = (m - 2 * raw.astype("f8")) / (m - p)
        initial = energy(base, base, u, q)
        rec = {}
        denom = 6 * (m - p) * D * D
        for k, v in zip(ARMS, (sync, colored), strict=True):
            off = int(np.count_nonzero((v != base) & ~band))
            assert off == 0
            path = out / f"sample{i}-{k}.npz"
            with path.open("xb") as fp:
                np.savez_compressed(fp, mask=v)
            rec[k] = dict(
                file=sha(path),
                mask=digest(v),
                additions=int(np.sum(v & ~base)),
                deletions=int(np.sum(~v & base)),
                off_support=off,
                cpu_gpu_differences=0,
                energy=energy(v, base, u, q),
            )
        change = sum(t["gain_numerator"] for t in trace) / (denom * base.size)
        observed = initial["total"] - rec["band-color"]["energy"]["total"]
        assert abs(change - observed) <= 1e-10 and observed >= 0
        spec[str(i)] = dict(
            raw=r["raw"],
            segmented=r["segmented"],
            source_crop=digest(raw),
            census=digest(base),
            phase_x2=phases,
            support=int(band.sum()),
            masks=rec,
            initial_energy=initial,
            exact_gain_denominator=denom,
            per_voxel_gain=change,
            gain_field_sha256=digest(cg),
            colors=trace,
        )
        print(
            i, observed, {k: v["additions"] + v["deletions"] for k, v in rec.items()}, flush=True
        )
    write(
        out / "predictions.json",
        dict(
            identity=ident,
            pipeline_id=token(ident),
            origin_xyz=[360, 376, 256],
            shape_zyx=list(SHAPE),
            specimens=spec,
        ),
    )


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions required")
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    predict(a.cache, a.out)
