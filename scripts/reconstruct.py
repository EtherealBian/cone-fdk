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
p.add_argument(
    "--planner",
    choices=["batch-first", "min-passes"],
    help="native scheduling policy; min-passes requires --roi-list, --stack or --cylinder",
)
group = p.add_mutually_exclusive_group()
group.add_argument("--roi", help="zero-based x,y,z,nx,ny,nz on the full --n grid; native only")
group.add_argument(
    "--roi-list",
    type=pathlib.Path,
    help="boxes.txt in parent-grid coordinates; write a CFDKROI1 pack",
)
group.add_argument(
    "--plane", type=pathlib.Path, help="CFDKPLANE1 physical slice; write raw [height,width] FP32"
)
group.add_argument(
    "--stack", type=pathlib.Path, help="CFDKSTACK1 parallel sections; [depth,height,width]"
)
group.add_argument(
    "--cylinder", type=pathlib.Path, help="CFDKCYL1 unwrapped sheets; [radius,height,azimuth]"
)
p.add_argument(
    "--stack-format", choices=["raw", "nrrd"], help="native --stack output; default raw FP32"
)
a = p.parse_args()
if a.stack_format and a.stack is None:
    p.error("--stack-format requires --stack")
if a.astra and a.planner is not None:
    p.error("--planner applies only to the native engine")
if a.astra and (
    a.roi is not None
    or a.roi_list is not None
    or a.plane is not None
    or a.stack is not None
    or a.cylinder is not None
):
    p.error(
        "region/plane/stack/cylinder selection applies to the native engine, not the ASTRA full-volume runner"
    )
print(
    json.dumps(
        astra_run(a.scan, a.output, a.n, a.fov)
        if a.astra
        else reconstruct(
            a.scan,
            a.output,
            a.n,
            a.fov,
            a.budget,
            a.batch,
            roi=a.roi,
            roi_list=a.roi_list,
            planner=a.planner,
            plane=a.plane,
            stack=a.stack,
            cylinder=a.cylinder,
            stack_format=a.stack_format,
        ),
        indent=2,
    )
)
