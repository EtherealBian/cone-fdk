import argparse
import json
import pathlib

from common import astra_run, reconstruct

p = argparse.ArgumentParser(description="Reconstruct full-circle flat-panel CT")
p.add_argument("scan", type=pathlib.Path, help="prefix of .geom and .f32")
p.add_argument("output", type=pathlib.Path)
p.add_argument("--n", type=int, default=256)
p.add_argument("--fov", type=float, required=True, help="isotropic physical field of view (mm)")
p.add_argument("--budget", type=int, default=768)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--astra", action="store_true")
a = p.parse_args()
print(
    json.dumps(
        astra_run(a.scan, a.output, a.n, a.fov)
        if a.astra
        else reconstruct(a.scan, a.output, a.n, a.fov, a.budget, a.batch),
        indent=2,
    )
)
