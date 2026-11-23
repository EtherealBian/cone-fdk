"""Fetch 8 original Zenodo archives with validated HTTP ranges and resumable chunks.
The frozen MD5s come from Zenodo record 6986012. No credentials required.
"""

import argparse
import concurrent.futures as cf
import hashlib
import os
import pathlib
import shutil
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIZES = [
    586938649,
    589322001,
    585348273,
    590007448,
    587828225,
    589492689,
    586804079,
    124490894,
]
MD5 = [
    "444ca44888869350f3691f2494880fd8",
    "772558ae698646abf54ad8a21a450e71",
    "4e9f9fb73411683988de5b5613d909a4",
    "05452e406cf2266077b723506c642608",
    "7614d7a1a1ec3be9137621d1cbc7543b",
    "2a8b2f59c2ffc42faeab2954c92a98ae",
    "b1ba468417cbecf52360e0f12da003fc",
    "0deabeccc10a524b141292d770180670",
]


def digest(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def fetch(jobs):
    base = ROOT / "data/downloads"
    base.mkdir(parents=True, exist_ok=True)
    cache = base / "chunks"
    cache.mkdir(exist_ok=True)
    chunk = 8 * 1024**2
    for i, (size, md) in enumerate(zip(SIZES, MD5), 1):
        name = f"20201111_walnut_projections_{i}.zip"
        dest = base / name
        if dest.exists() and dest.stat().st_size == size and digest(dest) == md:
            continue

        def part(j, i=i, size=size, name=name):
            start = j * chunk
            end = min(start + chunk, size) - 1
            p = cache / f"{i}-{j}"
            tmp = p.with_suffix(".part")
            header = p.with_suffix(".headers")
            if p.exists() and p.stat().st_size == end - start + 1:
                return p
            for trial in range(6):
                url = f"https://zenodo.org/api/records/6986012/files/{name}/content?part={j}&v={trial}"
                done = subprocess.run(
                    [
                        "curl.exe",
                        "-fsSL",
                        "--max-time",
                        "200",
                        "-D",
                        str(header),
                        "-r",
                        f"{start}-{end}",
                        "-o",
                        str(tmp),
                        url,
                    ],
                    capture_output=True,
                )
                if (
                    done.returncode == 0
                    and f"content-range: bytes {start}-{end}/{size}" in header.read_text().lower()
                    and tmp.stat().st_size == end - start + 1
                ):
                    tmp.replace(p)
                    return p
                time.sleep(4 + trial * 3)
            raise RuntimeError(f"failed range {i}/{j}: {done.stderr!r}")

        with cf.ThreadPoolExecutor(jobs) as pool:
            parts = list(pool.map(part, range((size + chunk - 1) // chunk)))
        tmp = dest.with_suffix(".verified-part")
        with tmp.open("wb") as out:
            for p in parts:
                with p.open("rb") as inp:
                    shutil.copyfileobj(inp, out)
        if digest(tmp) != md:
            raise RuntimeError("archive MD5 mismatch: " + name)
        os.replace(tmp, dest)
        print("verified", name, flush=True)
        for p in parts:
            p.unlink()
            p.with_suffix(".headers").unlink(missing_ok=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--jobs", type=int, default=6)
    fetch(a.parse_args().jobs)
