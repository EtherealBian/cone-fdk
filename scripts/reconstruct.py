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
group = p.add_mutually_exclusive_group()
group.add_argument("--roi", help="zero-based x,y,z,nx,ny,nz on the full --n grid; native only")
group.add_argument(
    "--roi-list",
    type=pathlib.Path,
    help="boxes.txt in parent-grid coordinates; write a CFDKROI1 pack",
)
a = p.parse_args()
if a.astra and (a.roi is not None or a.roi_list is not None):
    p.error("region selection applies to the native engine, not the ASTRA full-volume runner")
print(
    json.dumps(
        astra_run(a.scan, a.output, a.n, a.fov)
        if a.astra
        else reconstruct(
            a.scan, a.output, a.n, a.fov, a.budget, a.batch, roi=a.roi, roi_list=a.roi_list
        ),
        indent=2,
    )
)
