"""Build a separate sealed-input axis/remount/wall-pack overlay from frozen algorithms."""

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from build_edge_compat import one, sha
from sealed_inputs import publish_new

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = {
    "scripts/axis_gauge.py": "f93df4115479229e3c12ad5e3a200dddd6c97644538bc3499d1945450f7e18d5",
    "scripts/remount_loss.py": "79e61e1e6584148b16a4d141d659a1bd70c0142736baa221ecf6f2ca8d27f3a0",
    "scripts/wall_pack.py": "5ec5a9aecbf1c0e4b7ffe126b75d4b7cabf879d8b384ceb3c16e0be3c3b3ada6",
    "scripts/loss.py": "679d838afe271700f45e65f8deea93b18764ec205ab8e6ec3a2472bc01fe9a0a",
    "scripts/body_witness.py": "6575721414ea0c77a94d4d640978e63851c2cc0151f3add3afb8950f3ff493ac",
    "scripts/remount_datum.py": "2843247dff273f53dcf8f1df2b5c85058a05679bea8444005a3c3f7fabb1f885",
    "platform/spark/edge_job_inputs.py": "399e421fa886bc7638458a7e8e4fef036d5fdb23c032325edd4f826fac423272",
    "platform/spark/sealed_inputs.py": "16fc3e8b791d6aab57a204781e124b4d2eb8a1b98082db1e0392fddadffbdec2",
    "platform/spark/spark_publication.py": "a497b5387b56c602191da289b2dca36fbff9d58465a9c1e7060abca7a76dfa66",
}


def port(name, raw):
    s = raw.decode().replace("\r\n", "\n")
    if name == "axis_gauge.py":
        a = s.index("@contextmanager\ndef pinned_reader")
        b = s.index("    def fingerprints():", a)
        s = (
            s[:a]
            + """from measurement_jobs import pinned_reader, run_job


def run(scan, grid, job, *, fov, threshold, axis_limit, bore_limit, budget=96, batch=16, input_sha256=None):
    raw={"scan"+k:Path(scan).with_suffix(k) for k in (".geom",".f32")}
    raw["grid"]=grid
    def core(inputs, products, out, lease, opts):
        scan=inputs["scan.geom"].with_suffix("")
        sources={k:inputs["scan"+k] for k in (".geom",".f32")} | {k:products/k for k in ("cone-fdk.exe","kernels.ptx")}
        return _run_locked(scan, inputs["grid"], out, sources, fov, threshold, axis_limit, bore_limit, budget, batch, products)
    return run_job("axis",core,raw,job,BUILD,{},input_sha256)


def _run_locked(scan, grid, job, sources, fov, threshold, axis_limit, bore_limit, budget, batch, products):
"""
            + s[b:]
        )
        s = one(s, "batch, cylinder=spec)", "batch, cylinder=spec, build=products)")
        s = s.replace("budget=budget)", "budget=budget, build=products)")
    elif name == "loss.py":
        s = one(s, "*, budget=16):", "*, budget=16, build=BUILD):")
        s = s.replace("BUILD /", "build /")
    elif name == "remount_loss.py":
        s = one(
            s,
            "from axis_gauge import pinned_reader",
            "from measurement_jobs import DuplicateExposure, run_job",
        )
        s = one(s, "    witness=None,", "    witness=None,\n    input_sha256=None,")
        s = one(
            s, '    sources.update({n: BUILD / n for n in ["cone-fdk.exe", "kernels.ptx"]})\n', ""
        )
        a = s.index("    job.mkdir(")
        b = s.index("        for name, path in configs.items():", a)
        s = (
            s[:a]
            + """    def core(inputs, products, job, lease, opts):
        reference, inspected = (inputs[r+".geom"].with_suffix("") for r in ("reference","inspected"))
        sources = inputs | {n:products/n for n in ("cone-fdk.exe","kernels.ptx")}
        job.mkdir()
"""
            + s[b:]
        )
        s = one(
            s, "shutil.copyfile(path, job / name)", "shutil.copyfile(inputs[name], job / name)"
        )
        s = one(s, 'raise ValueError("two distinct', 'raise DuplicateExposure("two distinct')
        s = one(s, "budget, batch, roi=roi)", "budget, batch, roi=roi, build=products)")
        s = one(
            s, "budget, batch, cylinder=grid)", "budget, batch, cylinder=grid, build=products)"
        )
        s = one(s, "level, budget=budget)", "level, budget=budget, build=products)")
        s = one(
            s, "                budget=budget,", "                budget=budget, build=products,"
        )
        s = one(
            s,
            "        return log\n",
            '        return log\n    return run_job("remount",core,sources,job,BUILD,{},input_sha256)\n',
        )
    elif name == "wall_pack.py":
        s = one(s, "from axis_gauge import pinned_reader", "from measurement_jobs import run_job")
        a = s.index("def run(atlas")
        b = s.index("        pins = ", a)
        s = (
            s[:a]
            + """def run(atlas, grid, output, levels, *, budget=16, unpack=False, input_sha256=None):
    def core(inputs, products, out, lease, opts):
        atlas,grid = inputs['atlas'],inputs['grid']
        paths=inputs | {n:products/n for n in ('wall-pack.exe','kernels.ptx')}
"""
            + s[b:]
        )
        s = one(s, "pack, levels, budget=budget)", "pack, levels, budget=budget, build=products)")
        s = one(
            s,
            "        held.enter_context(pinned_reader(pack))",
            "        with lease.guard(pack):",
        )
        a = s.index("        n = check_stats")
        b = s.index("        temp = out", a)
        s = s[:a] + "".join("    " + line for line in s[a:b].splitlines(keepends=True)) + s[b:]
        s = one(
            s,
            "        return doc\n",
            '        return doc\n    return run_job("pack",core,{"atlas":atlas,"grid":grid},output,BUILD,{},input_sha256)\n',
        )
    else:
        return raw
    compile(s, name, "exec")
    return s.encode()


def build(out, expected=None):
    if sys.platform != "linux" or platform.machine() != "aarch64":
        raise ValueError("requires ARM64 Linux")
    out = Path(out).absolute()
    if out.parent != out.parent.resolve(strict=True) or out.name in ("", ".."):
        raise ValueError("real parent required")
    blobs = {}
    for rel, h in PINS.items():
        raw = (ROOT / rel).read_bytes()
        if sha(raw) != h:
            raise ValueError("frozen source changed: " + rel)
        name = Path(rel).name
        blobs[name] = port(name, raw)
    blobs["measurement_jobs.py"] = (HERE / "measurement_jobs.py").read_bytes()
    meta = dict(
        format="CFDK-Spark-measurement-overlay-v1",
        sources=PINS,
        generator_sha256=sha(Path(__file__).read_bytes()),
        products={k: sha(v) for k, v in blobs.items()},
    )
    blobs["overlay.json"] = (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode()
    pin = sha(blobs["overlay.json"])
    if expected is not None and pin != expected:
        raise ValueError("unexpected overlay digest")
    if os.path.lexists(out):
        if (
            out.is_symlink()
            or not out.is_dir()
            or set(blobs) != {p.name for p in out.iterdir()}
            or any(
                (out / k).is_symlink() or not (out / k).is_file() or (out / k).read_bytes() != v
                for k, v in blobs.items()
            )
        ):
            raise ValueError("existing overlay not identical")
    else:
        with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
            stage = Path(tmp) / "stage"
            stage.mkdir()
            for k, v in blobs.items():
                (stage / k).write_bytes(v)
            publish_new(stage, out)
    return pin


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out", type=Path)
    cli.add_argument("--expect")
    a = cli.parse_args()
    print(build(a.out, a.expect))
