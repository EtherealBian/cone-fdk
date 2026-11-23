"""Exercise a clean extraction with no project data or generated outputs (Windows)."""

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile

from export_volume import verify_viewer
from release import verify


def run(bundle):
    bundle = bundle.resolve()

    with bundle.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    pub = bundle.with_suffix(bundle.suffix + ".sha256").read_text().split()[0]
    if sha != pub:
        raise RuntimeError("archive checksum mismatch")
    with tempfile.TemporaryDirectory() as tmp:
        dst = pathlib.Path(tmp) / "ConeFDK 离线复验"
        dst.mkdir()
        with zipfile.ZipFile(bundle) as z:
            for n in z.namelist():
                if not (dst / n).resolve().is_relative_to(dst):
                    raise ValueError("unsafe release path")
            bad = z.testzip()
            if bad:
                raise ValueError("CRC mismatch: " + bad)
            z.extractall(dst)
        verify(dst)
        if (dst / "data").exists() or (dst / ".venv").exists():
            raise ValueError("not a clean package")
        reports = dst / "reports"
        gold = json.loads((reports / "walnut.json").read_text())
        viewer = reports / "walnut-viewer.html"
        for name in ["walnut.json", "walnut.png", "walnut-orthogonal.png", "walnut-viewer.html"]:
            (reports / name).unlink()
        shell = shutil.which("pwsh") or shutil.which("powershell")
        args = [shell, "-NoProfile", "-File", str(dst / "Demo.ps1"), "-Python", sys.executable]
        subprocess.run(args, cwd=dst, check=True)
        new = json.loads((reports / "walnut.json").read_text())
        rel = new["corrected_vs_astra"]["rel_l2"]
        if new["views"] != 90 or not rel < 0.02:
            raise AssertionError("numerical demo gate")
        vol = dst / "data/walnut-corrected-256.f32"
        header = vol.with_suffix(".mhd")
        if (
            vol.stat().st_size != 4 * 256**3
            or ("ElementDataFile = " + vol.name) not in header.read_text()
        ):
            raise AssertionError("MetaImage output")
        preview_count = verify_viewer(viewer, 256)
        full_preview_count = verify_viewer(reports / "full/walnut-viewer.html", 512)
        with vol.open("rb") as f:
            outsha = hashlib.file_digest(f, "sha256").hexdigest()
        if outsha != gold["corrected_sha256"] or outsha != new["corrected_sha256"]:
            raise AssertionError("clean demo output differs from release-qualified volume")
        record = dict(
            matches_release_volume=True,
            bundle_sha256=sha,
            all_members_verified=True,
            clean_extraction=True,
            unicode_path=True,
            used_separate_existing_venv=True,
            views=90,
            preview_slices_decoded=preview_count,
            full_preview_slices_decoded=full_preview_count,
            volume_sha256=outsha,
            rel_l2=rel,
            release_recorded_rel_l2=gold["corrected_vs_astra"]["rel_l2"],
        )
        (bundle.parent / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=pathlib.Path)
    run(p.parse_args().archive)
