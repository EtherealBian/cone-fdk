"""Seal baseline probabilities on all transfer slabs without opening their annotations."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pores"))
import rank_model  # noqa: E402
from denoise import Denoise  # noqa: E402
from freeze import DEV, identity, token  # noqa: E402
from local_model import SETTINGS, median_stack  # noqa: E402
from neck_model import HALO, Z_HALO  # noqa: E402
from residual_model import probability  # noqa: E402
from validate import sha, stack, verified  # noqa: E402

Z = (256, 352)
SHAPE = (Z[1] - Z[0], 256, 256)


def blocks(raw, i, begin, end, single, ranked):
    tail = None
    h = 30 / SETTINGS[i][0]
    for a in range(begin, end, 4):
        b = min(a + 4, end)
        left = a - Z_HALO if tail is None else a + Z_HALO
        med = median_stack(raw, i, left, b + Z_HALO, HALO)
        fields = np.concatenate((single(med, h), ranked(med, h)), axis=1)
        if tail is not None:
            med = np.concatenate((tail[0], med))
            fields = np.concatenate((tail[1], fields))
        tail = med[-4:].copy(), fields[-4:].copy()
        yield a, b, rank_model.features(fields, med, i)


def generate(cache, out, z=Z):
    if tuple(z) not in (Z, (552, 680)):
        raise ValueError("Unregistered development slab")
    shape = (z[1] - z[0], *SHAPE[1:])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new empty directory for blind baseline predictions")
    ident = token(identity("E"))
    records = {}
    with (
        Denoise(ROOT / "build/pores/adaptive.ptx", adaptive=True) as single,
        Denoise(ROOT / "build/pores/rankpatch.ptx", ranked=True) as ranked,
    ):
        for i in DEV:
            published = json.loads(
                (ROOT / f"pores/evidence/candidate-e/sample{i}.json").read_text()
            )
            if published["pipeline_id"] != ident:
                raise ValueError("Rebuild E in the locked environment")
            raw, rs = verified(cache, i, "raw", False)
            seg, ss = verified(cache, i, "segmented", False)
            print(i, "fitting original E on z=160:184", flush=True)
            model, cut = rank_model.train(
                [
                    (x, mask, stack(seg, i, "segmented", a, b))
                    for a, b, (x, mask) in blocks(raw, i, 160, 184, single, ranked)
                ]
            )
            if cut != 0.5:
                raise ValueError("E baseline changed")
            path = out / f"sample{i}.f32"
            with path.open("xb") as fp:
                for a, b, (x, mask) in blocks(raw, i, *z, single, ranked):
                    p = probability(model, x, mask)
                    if p.shape != (b - a, *SHAPE[1:]) or not np.isfinite(p).all():
                        raise ValueError("Invalid probabilities")
                    p.astype("<f4", copy=False).tofile(fp)
            records[str(i)] = dict(
                raw=rs["sha256"], segmented=ss["sha256"], probability=sha(path)
            )
            print(i, records[str(i)]["probability"], flush=True)
    manifest = dict(schema=1, model=ident, z=z, shape_zyx=shape, dtype="<f4", specimens=records)
    (out / "baseline.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split-pilot", action="store_true", help="Use the registered z=552:680 development slab")
    a = ap.parse_args()
    generate(a.cache, a.out, (552, 680) if a.split_pilot else Z)
