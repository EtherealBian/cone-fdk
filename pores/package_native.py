"""Package the mask-only native measurement preview with a deterministic Windows smoke test."""

import argparse
import hashlib
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from pore_check import ROOT, native, reference


def package(output):
    shape = (28, 48, 64)
    a = np.ones(shape, "<f4")
    a[3:8, 6:10, 6:12] = 0
    a[13:21, 25:32, 35:44] = 0
    a[:, 37:39, 49:52] = 0
    spacing = (0.75, 1.25, 2.5)
    dist = ROOT / "build/pores"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        td = Path(tmp)
        src = td / "fixture.f32"
        a.tofile(src)
        table = td / "expected.csv"
        st = native(src, table, shape, spacing, 0, 6, 3)
        _, rows = reference(a == 0, 6, spacing, table, st)
        if sorted(int(r["voxels"]) for r in rows) != [120, 168, 504] or st["spanning_xyz"] != [
            0,
            0,
            1,
        ]:
            raise ValueError("Fixture agreement failed")

        def dh(b):
            return hashlib.sha256(b).hexdigest()

        script = """$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    $out=[System.IO.Path]::GetTempFileName()
    try {
        & .\\cone-pores.exe --input fixture.f32 --output $out --nx 64 --ny 48 --nz 28 --sx .75 --sy 1.25 --sz 2.5 --threshold 0 --filter none --connectivity 6 --slab 3 --budget-mib 4
        if($LASTEXITCODE -ne 0){throw 'Native executable failed'}
        if([Convert]::ToBase64String([IO.File]::ReadAllBytes($out)) -cne [Convert]::ToBase64String([IO.File]::ReadAllBytes((Join-Path $PSScriptRoot 'expected.csv')))){throw 'Object table mismatch'}
        Write-Host 'PASS: three known discrete objects; complete, bit-identical component table.'
    } finally { if(Test-Path -LiteralPath $out){Remove-Item -LiteralPath $out} }
} finally {Pop-Location}
"""
        data = {n: (dist / n).read_bytes() for n in ["cone-pores.exe", "pores.ptx"]}
        data.update(
            {
                "fixture.f32": src.read_bytes(),
                "expected.csv": table.read_bytes(),
                "Run-Demo.ps1": script.encode("ascii"),
                "LICENSE": (ROOT / "LICENSE").read_bytes(),
                "NOTICE.md": (ROOT / "NOTICE.md").read_bytes(),
                "README.txt": b"EXPERIMENTAL / known-mask metrology only. Not a certified automatic raw-to-mask inspection system. Windows x64 with compatible NVIDIA GPU/driver, no Python needed. Unzip all files together and run .\\Run-Demo.ps1 from PowerShell. The synthetic anisotropic fixture has three separated voids (120, 168, 504 voxels); the 168-voxel channel spans both Z faces. Expected CSV includes all rows, including this clipped channel. See the measurement-lab branch README for the input format and units.\n",
            }
        )
        data["SHA256SUMS.txt"] = "".join(f"{dh(data[n])}  {n}\n" for n in sorted(data)).encode(
            "ascii"
        )
        with ZipFile(output, "w") as z:
            for n in sorted(data):
                info = ZipInfo(n, (2026, 11, 23, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 0
                z.writestr(info, data[n])
        with ZipFile(output) as z:
            folder = td / "clean extract"
            z.extractall(folder)
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(folder / "Run-Demo.ps1"),
            ],
            cwd=folder,
            check=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path)
    package(ap.parse_args().output)
