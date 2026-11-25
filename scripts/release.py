"""Create a portable source+binary release, or verify an extracted release (stdlib only)."""

import argparse
import hashlib
import json
import os
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(base):
    base = pathlib.Path(base).resolve()
    entries = (base / "SHA256SUMS").read_text(encoding="utf8").splitlines()
    if not entries:
        raise ValueError("empty manifest")
    for line in entries:
        expected, name = line.split("  ", 1)
        p = (base / name).resolve()
        if not p.is_relative_to(base) or digest(p) != expected:
            raise ValueError("invalid release member: " + name)
    print(f"{len(entries)} release members verified against SHA256SUMS", flush=True)


def build():
    # Never ship an optimization qualification tied to a different executable/PTX.
    evidence = ROOT / "reports"
    measured = json.loads((evidence / "native-ab.json").read_text())
    repeated = json.loads((evidence / "run-repeat.json").read_text())
    built = json.loads((evidence / "build-repro.json").read_text())
    frozen = {k: digest(ROOT / "build" / k) for k in ("cone-fdk.exe", "kernels.ptx")}
    if (
        measured["pilot"]
        or not measured["passed"]
        or measured["binaries"]["new"] != frozen
        or not repeated["identical"]
        or repeated["binaries"] != measured["binaries"]
        or not built["identical"]
        or built["executable_sha256"] != frozen["cone-fdk.exe"]
        or built["ptx_sha256"] != frozen["kernels.ptx"]
    ):
        raise RuntimeError("stale or missing build / performance / output evidence")
    # Deliberate allowlist: no driver/toolchain DLLs, venv, raw data or download cache.
    files = [
        ROOT / n
        for n in (
            "README.md",
            "README.zh-CN.md",
            "LICENSE",
            "NOTICE.md",
            ".clang-format",
            ".gitignore",
            "Build.ps1",
            "Demo.ps1",
            "Run-Acceptance.ps1",
            "CMakeLists.txt",
            "requirements-win.lock",
            "pyproject.toml",
            "build/cone-fdk.exe",
            "build/kernels.ptx",
        )
    ]
    for name, patterns in [
        ("src", ["*.cu", "*.cpp", "*.hpp"]),
        ("scripts", ["*.py", "cylinder_atlas.html"]),
        ("tests", ["*.py"]),
        ("fixtures", ["*.npz", "*.geom", "*.json", "*.txt"]),
        ("reports", ["*.md", "*.json", "*.png", "*.gif", "*.html", "*.xml", "*.txt"]),
    ]:
        for pat in patterns:
            files.extend((ROOT / name).rglob(pat))
    files.append(evidence / "wall-loss-demo.zip")
    for required in (
        "reports/RESULTS.md",
        "reports/OPTIMIZATION.md",
        "reports/full/walnut-viewer.html",
        "reports/walnut-full.json",
        "fixtures/walnut-s8-b8/source-scan.txt",
    ):
        if ROOT / required not in files:
            raise RuntimeError(f"missing release input: {required}")
    dest = ROOT / "dist"
    dest.mkdir(exist_ok=True)
    archive = dest / "ConeFDK-win-x64.zip"
    tmp = archive.with_suffix(".zip.part")
    lines = []
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            rel = p.relative_to(ROOT).as_posix()
            lines.append(f"{digest(p)}  {rel}")
            z.write(p, rel)
        z.writestr("SHA256SUMS", "\n".join(lines) + "\n")
    with zipfile.ZipFile(tmp) as z:
        if bad := z.testzip():
            raise RuntimeError("ZIP CRC mismatch: " + bad)
    os.replace(tmp, archive)
    checksum = digest(archive)
    (dest / (archive.name + ".sha256")).write_text(checksum + "  " + archive.name + "\n")
    print(archive, archive.stat().st_size, checksum)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["build", "verify"])
    p.add_argument("directory", nargs="?", type=pathlib.Path, default=ROOT)
    a = p.parse_args()
    build() if a.action == "build" else verify(a.directory)
